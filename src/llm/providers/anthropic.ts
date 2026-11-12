import type { LLMProvider, Model, Message, ToolDef, StreamChunk, ToolCall, ChatOptions } from '../provider';

interface AnthropicContentBlock {
  type: 'text' | 'tool_use';
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export class AnthropicProvider implements LLMProvider {
  id = 'anthropic';
  name = 'Anthropic';

  private apiKey = '';
  private model = 'claude-sonnet-4-20250514';
  private baseUrl = 'https://api.anthropic.com';

  configSchema: LLMProvider['configSchema'] = {
    fields: [
      {
        key: 'apiKey',
        label: 'API Key',
        type: 'password',
        required: true,
        placeholder: 'sk-ant-...',
        helpText: 'Your Anthropic API key from console.anthropic.com',
      },
      {
        key: 'model',
        label: 'Model',
        type: 'select',
        required: true,
        default: 'claude-sonnet-4-20250514',
        options: [
          { label: 'Claude Opus 4', value: 'claude-opus-4-20250514' },
          { label: 'Claude Sonnet 4', value: 'claude-sonnet-4-20250514' },
          { label: 'Claude 3.5 Haiku', value: 'claude-3-5-haiku-20241022' },
          { label: 'Claude 3 Opus', value: 'claude-3-opus-20240229' },
          { label: 'Claude 3 Haiku', value: 'claude-3-haiku-20240307' },
        ],
      },
    ],
    testPrompt: 'Say "hello" in one word.',
  };

  configure(config: Record<string, string>): void {
    this.apiKey = config.apiKey || '';
    this.model = config.model || 'claude-sonnet-4-20250514';
  }

  /**
   * Convert OpenAI-style messages to Anthropic format.
   * Anthropic requires a separate system parameter and specific content block formatting.
   */
  private convertMessages(messages: Message[]): {
    system: string | undefined;
    messages: Array<{
      role: 'user' | 'assistant';
      content: string | Array<Record<string, unknown>>;
    }>;
  } {
    let system: string | undefined;
    const converted: Array<{
      role: 'user' | 'assistant';
      content: string | Array<Record<string, unknown>>;
    }> = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        system = (system ? system + '\n' : '') + msg.content;
        continue;
      }

      if (msg.role === 'tool') {
        // Tool results in Anthropic are sent as user messages with tool_result content blocks.
        // Merge consecutive tool results into a single user message to satisfy Anthropic's
        // strict alternating user/assistant turn requirement.
        const block = {
          type: 'tool_result',
          tool_use_id: msg.tool_call_id,
          content: msg.content,
        };
        const last = converted[converted.length - 1];
        if (last && last.role === 'user' && Array.isArray(last.content)) {
          last.content.push(block);
        } else {
          converted.push({ role: 'user', content: [block] });
        }
        continue;
      }

      if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
        // Assistant message with tool calls
        const content: Array<Record<string, unknown>> = [];
        if (msg.content) {
          content.push({ type: 'text', text: msg.content });
        }
        for (const tc of msg.tool_calls) {
          let input: Record<string, unknown> = {};
          try {
            input = JSON.parse(tc.function.arguments || '{}');
          } catch {
            // keep empty object
          }
          content.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
            input,
          });
        }
        converted.push({ role: 'assistant', content });
        continue;
      }

      const role = msg.role === 'user' ? 'user' : 'assistant';

      // Anthropic doesn't allow consecutive messages with the same role.
      // Merge if needed.
      const last = converted[converted.length - 1];
      if (last && last.role === role) {
        if (typeof last.content === 'string' && typeof msg.content === 'string') {
          last.content += '\n' + msg.content;
        } else if (Array.isArray(last.content)) {
          last.content.push({ type: 'text', text: msg.content });
        } else {
          converted.push({ role, content: msg.content });
        }
      } else {
        converted.push({ role, content: msg.content });
      }
    }

    // Anthropic requires messages to start with a user message
    if (converted.length > 0 && converted[0].role !== 'user') {
      converted.unshift({ role: 'user', content: 'Hello.' });
    }

    return { system, messages: converted };
  }

  private convertTools(tools: ToolDef[]): AnthropicTool[] {
    return tools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));
  }

  async *chat(messages: Message[], tools?: ToolDef[], options?: ChatOptions): AsyncGenerator<StreamChunk> {
    if (!this.apiKey) {
      yield { type: 'error', error: 'Anthropic API key is not configured' };
      return;
    }

    const { system, messages: anthropicMessages } = this.convertMessages(messages);

    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: options?.maxTokens ?? 8192,
      messages: anthropicMessages,
      stream: true,
    };
    if (options?.temperature !== undefined) body.temperature = options.temperature;
    if (system) body.system = system;
    if (tools && tools.length > 0) body.tools = this.convertTools(tools);

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120_000),
      });
    } catch (err: any) {
      yield { type: 'error', error: `Anthropic request failed: ${err.message}` };
      return;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      yield { type: 'error', error: `Anthropic API error ${response.status}: ${errorText}` };
      return;
    }

    // Parse Anthropic SSE stream
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const activeTools = new Map<number, { id: string; name: string; arguments: string }>();
    let promptTokens = 0;
    let completionTokens = 0;
    let emittedDone = false;

    const processLine = function* (line: string): Generator<StreamChunk | { __stopError?: boolean }> {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(':') || trimmed.startsWith('event:')) return;
      if (!trimmed.startsWith('data:')) return;

      const data = trimmed.slice(5).trimStart();
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        return;
      }

      switch (parsed.type) {
        case 'message_start':
          if (parsed.message?.usage) {
            promptTokens = parsed.message.usage.input_tokens || 0;
          }
          break;

        case 'content_block_start':
          if (parsed.content_block?.type === 'tool_use') {
            const idx = parsed.index ?? 0;
            activeTools.set(idx, {
              id: parsed.content_block.id || '',
              name: parsed.content_block.name || '',
              arguments: '',
            });
          }
          break;

        case 'content_block_delta':
          if (parsed.delta?.type === 'text_delta' && parsed.delta.text) {
            yield { type: 'text', text: parsed.delta.text };
          } else if (parsed.delta?.type === 'input_json_delta') {
            const idx = parsed.index ?? 0;
            const active = activeTools.get(idx);
            if (active) {
              active.arguments += parsed.delta.partial_json || '';
            }
          }
          break;

        case 'content_block_stop': {
          const idx = parsed.index ?? 0;
          const active = activeTools.get(idx);
          if (active) {
            yield {
              type: 'tool_call',
              toolCall: {
                id: active.id,
                type: 'function',
                function: {
                  name: active.name,
                  arguments: active.arguments.trim() || '{}',
                },
              },
            };
            activeTools.delete(idx);
          }
          break;
        }

        case 'message_delta':
          if (parsed.usage) {
            completionTokens = parsed.usage.output_tokens || 0;
          }
          break;

        case 'message_stop':
          emittedDone = true;
          yield {
            type: 'done',
            usage: { promptTokens, completionTokens },
          };
          break;

        case 'error':
          yield {
            type: 'error',
            error: parsed.error?.message || 'Unknown Anthropic stream error',
          };
          yield { __stopError: true };
          break;
      }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          for (const out of processLine(line)) {
            if ('__stopError' in out) return;
            yield out;
          }
        }
      }

      buffer += decoder.decode();
      if (buffer.trim()) {
        for (const line of buffer.split('\n')) {
          for (const out of processLine(line)) {
            if ('__stopError' in out) return;
            yield out;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Flush any unclosed tool block
    for (const [, active] of activeTools) {
      yield {
        type: 'tool_call',
        toolCall: {
          id: active.id,
          type: 'function',
          function: {
            name: active.name,
            arguments: active.arguments.trim() || '{}',
          },
        },
      };
    }

    if (!emittedDone) {
      yield { type: 'done', usage: { promptTokens, completionTokens } };
    }
  }

  async listModels(): Promise<Model[]> {
    // Anthropic doesn't have a public models listing endpoint, return known models
    return [
      { id: 'claude-opus-4-20250514', name: 'Claude Opus 4', contextLength: 200000, supportsTools: true },
      { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4', contextLength: 200000, supportsTools: true },
      { id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku', contextLength: 200000, supportsTools: true },
      { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', contextLength: 200000, supportsTools: true },
      { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', contextLength: 200000, supportsTools: true },
    ];
  }

  supportsToolUse(): boolean {
    return true;
  }

  supportsStreaming(): boolean {
    return true;
  }
}
