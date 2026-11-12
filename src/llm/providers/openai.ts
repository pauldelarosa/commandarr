import type { LLMProvider, ConfigField, Model, Message, ToolDef, StreamChunk, ChatOptions } from '../provider';
import { nanoid } from 'nanoid';

const DEFAULT_LLM_TIMEOUT_MS = 120_000;

function normalizeToolArgs(rawArgs?: string): string {
  const trimmed = rawArgs?.trim();
  return trimmed ? trimmed : '{}';
}

/**
 * Format internal messages into standard OpenAI chat completion payloads.
 */
export function formatOpenAIMessages(messages: Message[]): Record<string, unknown>[] {
  return messages.map((m) => {
    const msg: Record<string, unknown> = { role: m.role, content: m.content };
    if (m.name) msg.name = m.name;
    if (m.tool_calls && m.tool_calls.length > 0) {
      msg.tool_calls = m.tool_calls.map((tc) => ({
        id: tc.id,
        type: tc.type,
        function: {
          name: tc.function.name,
          arguments: normalizeToolArgs(tc.function.arguments),
        },
      }));
    }
    if (m.tool_call_id) msg.tool_call_id = m.tool_call_id;
    return msg;
  });
}

/**
 * Parse an SSE stream from an OpenAI-compatible API response.
 * Yields parsed JSON objects for each `data:` line (except `[DONE]`).
 */
async function* parseSSE(response: Response): AsyncGenerator<any> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const parseLine = (line: string): { done: boolean; value?: any } => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(':')) return { done: false };
    if (trimmed.startsWith('data:')) {
      const data = trimmed.slice(5).trimStart();
      if (data === '[DONE]') return { done: true };
      try {
        return { done: false, value: JSON.parse(data) };
      } catch {
        // skip unparseable lines
      }
    }
    return { done: false };
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const parsed = parseLine(line);
        if (parsed.done) return;
        if (parsed.value !== undefined) yield parsed.value;
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      for (const line of buffer.split('\n')) {
        const parsed = parseLine(line);
        if (parsed.done) return;
        if (parsed.value !== undefined) yield parsed.value;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Consume an OpenAI-compatible SSE stream and yield normalized StreamChunks.
 */
export async function* streamOpenAICompatibleChat(
  response: Response,
  providerLabel: string,
): AsyncGenerator<StreamChunk> {
  const pendingToolCalls = new Map<number, { id: string; name: string; arguments: string }>();
  let usage: StreamChunk['usage'] | undefined;

  for await (const chunk of parseSSE(response)) {
    if (chunk.error) {
      const errMsg = typeof chunk.error === 'string'
        ? chunk.error
        : chunk.error?.message || JSON.stringify(chunk.error);
      yield { type: 'error', error: `${providerLabel} stream error: ${errMsg}` };
      return;
    }

    if (chunk.usage) {
      usage = {
        promptTokens: chunk.usage.prompt_tokens || 0,
        completionTokens: chunk.usage.completion_tokens || 0,
      };
    }

    const choice = chunk.choices?.[0];
    if (!choice) continue;

    const delta = choice.delta;
    if (delta) {
      if (delta.content) {
        yield { type: 'text', text: delta.content };
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          const existing = pendingToolCalls.get(idx);
          if (!existing) {
            pendingToolCalls.set(idx, {
              id: tc.id || `call_${nanoid(10)}`,
              name: tc.function?.name || '',
              arguments: tc.function?.arguments || '',
            });
          } else {
            if (tc.id && !existing.id) existing.id = tc.id;
            if (tc.function?.name) existing.name += tc.function.name;
            if (tc.function?.arguments) existing.arguments += tc.function.arguments;
          }
        }
      }
    }

    const finishReason = choice.finish_reason;
    if (finishReason === 'content_filter') {
      yield { type: 'error', error: `${providerLabel} response was filtered by content policy.` };
      return;
    }

    if (
      finishReason === 'tool_calls' ||
      finishReason === 'stop' ||
      finishReason === 'length' ||
      finishReason === 'function_call'
    ) {
      for (const [, tc] of pendingToolCalls) {
        yield {
          type: 'tool_call',
          toolCall: {
            id: tc.id,
            type: 'function',
            function: { name: tc.name, arguments: normalizeToolArgs(tc.arguments) },
          },
        };
      }
      pendingToolCalls.clear();
    }
  }

  // Flush any tool calls if stream ended without a finish_reason chunk
  for (const [, tc] of pendingToolCalls) {
    yield {
      type: 'tool_call',
      toolCall: {
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: normalizeToolArgs(tc.arguments) },
      },
    };
  }

  yield { type: 'done', usage };
}

export class OpenAIProvider implements LLMProvider {
  id = 'openai';
  name = 'OpenAI';

  private apiKey = '';
  private model = 'gpt-4o';
  private orgId = '';
  private baseUrl = 'https://api.openai.com/v1';

  configSchema: LLMProvider['configSchema'] = {
    fields: [
      {
        key: 'apiKey',
        label: 'API Key',
        type: 'password',
        required: true,
        placeholder: 'sk-...',
        helpText: 'Your OpenAI API key from platform.openai.com',
      },
      {
        key: 'model',
        label: 'Model',
        type: 'select',
        required: true,
        default: 'gpt-4o',
        options: [
          { label: 'GPT-4o', value: 'gpt-4o' },
          { label: 'GPT-4o Mini', value: 'gpt-4o-mini' },
          { label: 'GPT-4 Turbo', value: 'gpt-4-turbo' },
          { label: 'GPT-4', value: 'gpt-4' },
          { label: 'GPT-3.5 Turbo', value: 'gpt-3.5-turbo' },
          { label: 'o1', value: 'o1' },
          { label: 'o1 Mini', value: 'o1-mini' },
          { label: 'o3 Mini', value: 'o3-mini' },
        ],
      },
      {
        key: 'orgId',
        label: 'Organization ID',
        type: 'text',
        required: false,
        placeholder: 'org-...',
        helpText: 'Optional OpenAI organization ID',
      },
    ],
    testPrompt: 'Say "hello" in one word.',
  };

  configure(config: Record<string, string>): void {
    this.apiKey = config.apiKey || '';
    this.model = config.model || 'gpt-4o';
    this.orgId = config.orgId || '';
    if (config.baseUrl) this.baseUrl = config.baseUrl;
  }

  async *chat(messages: Message[], tools?: ToolDef[], options?: ChatOptions): AsyncGenerator<StreamChunk> {
    if (!this.apiKey) {
      yield { type: 'error', error: 'OpenAI API key is not configured' };
      return;
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
    };
    if (this.orgId) headers['OpenAI-Organization'] = this.orgId;

    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: options?.maxTokens ?? 4096,
      messages: formatOpenAIMessages(messages),
      stream: true,
    };
    if (options?.temperature !== undefined) body.temperature = options.temperature;
    if (tools && tools.length > 0) body.tools = tools;

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(DEFAULT_LLM_TIMEOUT_MS),
      });
    } catch (err: any) {
      yield { type: 'error', error: `OpenAI request failed: ${err.message}` };
      return;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      yield { type: 'error', error: `OpenAI API error ${response.status}: ${errorText}` };
      return;
    }

    yield* streamOpenAICompatibleChat(response, 'OpenAI');
  }

  async listModels(): Promise<Model[]> {
    if (!this.apiKey) return [];

    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.apiKey}`,
    };
    if (this.orgId) headers['OpenAI-Organization'] = this.orgId;

    try {
      const response = await fetch(`${this.baseUrl}/models`, { headers });
      if (!response.ok) return [];

      const data = await response.json() as { data?: { id: string; owned_by?: string }[] };
      return (data.data || [])
        .filter((m) => m.id.startsWith('gpt-') || m.id.startsWith('o1') || m.id.startsWith('o3'))
        .map((m) => ({
          id: m.id,
          name: m.id,
          supportsTools: !m.id.includes('instruct'),
        }))
        .sort((a, b) => a.id.localeCompare(b.id));
    } catch {
      return [];
    }
  }

  supportsToolUse(): boolean {
    return true;
  }

  supportsStreaming(): boolean {
    return true;
  }
}

export { parseSSE };
