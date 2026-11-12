import type { LLMProvider, Model, Message, ToolDef, StreamChunk, ChatOptions } from '../provider';
import { nanoid } from 'nanoid';

interface GeminiContent {
  role: 'user' | 'model';
  parts: GeminiPart[];
}

type GeminiPart =
  | { text: string; thought?: boolean }
  | { functionCall: { name: string; args: Record<string, unknown> }; thoughtSignature?: string }
  | { functionResponse: { name: string; response: { result: unknown } } };

interface GeminiToolDeclaration {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export class GoogleProvider implements LLMProvider {
  id = 'google';
  name = 'Google Gemini';

  private apiKey = '';
  private model = 'gemini-2.0-flash';
  private baseUrl = 'https://generativelanguage.googleapis.com/v1beta';

  configSchema: LLMProvider['configSchema'] = {
    fields: [
      {
        key: 'apiKey',
        label: 'API Key',
        type: 'password',
        required: true,
        placeholder: 'AI...',
        helpText: 'Your Google AI API key from aistudio.google.com',
      },
      {
        key: 'model',
        label: 'Model',
        type: 'select',
        required: true,
        default: 'gemini-2.0-flash',
        options: [
          { label: 'Gemini 2.0 Flash', value: 'gemini-2.0-flash' },
          { label: 'Gemini 2.0 Flash Lite', value: 'gemini-2.0-flash-lite' },
          { label: 'Gemini 1.5 Pro', value: 'gemini-1.5-pro' },
          { label: 'Gemini 1.5 Flash', value: 'gemini-1.5-flash' },
          { label: 'Gemini 1.5 Flash 8B', value: 'gemini-1.5-flash-8b' },
        ],
      },
    ],
    testPrompt: 'Say "hello" in one word.',
  };

  configure(config: Record<string, string>): void {
    this.apiKey = config.apiKey || '';
    this.model = config.model || 'gemini-2.0-flash';
  }

  /**
   * Convert OpenAI-style messages to Gemini format.
   */
  private convertMessages(messages: Message[]): {
    systemInstruction: { parts: { text: string }[] } | undefined;
    contents: GeminiContent[];
  } {
    let systemText = '';
    const contents: GeminiContent[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemText += (systemText ? '\n' : '') + msg.content;
        continue;
      }

      if (msg.role === 'tool') {
        // Tool response goes as a user message with functionResponse part.
        // Merge consecutive tool responses into one user message for parallel calls.
        let result: unknown;
        try {
          result = JSON.parse(msg.content);
        } catch {
          result = msg.content;
        }
        const responsePart: GeminiPart = {
          functionResponse: {
            name: msg.name || 'unknown',
            response: { result },
          },
        };
        const last = contents[contents.length - 1];
        if (last && last.role === 'user' && last.parts.some((p) => 'functionResponse' in p)) {
          last.parts.push(responsePart);
        } else {
          contents.push({
            role: 'user',
            parts: [responsePart],
          });
        }
        continue;
      }

      if (msg.role === 'assistant') {
        const parts: GeminiPart[] = [];
        if (msg.content) {
          parts.push({ text: msg.content });
        }
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            let args: Record<string, unknown> = {};
            try {
              args = JSON.parse(tc.function.arguments || '{}');
            } catch {
              // keep empty
            }
            const fcPart: GeminiPart = {
              functionCall: { name: tc.function.name, args },
            };
            if (tc.thoughtSignature) {
              fcPart.thoughtSignature = tc.thoughtSignature;
            }
            parts.push(fcPart);
          }
        }
        if (parts.length > 0) {
          contents.push({ role: 'model', parts });
        }
        continue;
      }

      // User messages
      const role = 'user';
      // Merge consecutive user messages
      const last = contents[contents.length - 1];
      if (last && last.role === 'user' && last.parts.length === 1 && 'text' in last.parts[0]) {
        (last.parts[0] as { text: string }).text += '\n' + msg.content;
      } else {
        contents.push({ role, parts: [{ text: msg.content }] });
      }
    }

    return {
      systemInstruction: systemText ? { parts: [{ text: systemText }] } : undefined,
      contents,
    };
  }

  private convertTools(tools: ToolDef[]): { functionDeclarations: GeminiToolDeclaration[] }[] {
    return [
      {
        functionDeclarations: tools.map((t) => ({
          name: t.function.name,
          description: t.function.description,
          parameters: t.function.parameters,
        })),
      },
    ];
  }

  async *chat(messages: Message[], tools?: ToolDef[], options?: ChatOptions): AsyncGenerator<StreamChunk> {
    if (!this.apiKey) {
      yield { type: 'error', error: 'Google API key is not configured' };
      return;
    }

    const { systemInstruction, contents } = this.convertMessages(messages);

    const body: Record<string, unknown> = { contents };
    const generationConfig: Record<string, unknown> = {
      maxOutputTokens: options?.maxTokens ?? 8192,
    };
    if (options?.temperature !== undefined) generationConfig.temperature = options.temperature;
    body.generationConfig = generationConfig;
    if (systemInstruction) body.systemInstruction = systemInstruction;
    if (tools && tools.length > 0) body.tools = this.convertTools(tools);

    const url = `${this.baseUrl}/models/${this.model}:streamGenerateContent?alt=sse&key=${this.apiKey}`;

    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120_000),
      });
    } catch (err: any) {
      yield { type: 'error', error: `Google API request failed: ${err.message}` };
      return;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      yield { type: 'error', error: `Google API error ${response.status}: ${errorText}` };
      return;
    }

    // Parse Gemini SSE stream
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let promptTokens = 0;
    let completionTokens = 0;
    let hasEmittedContent = false;
    let lastFinishReason: string | undefined;

    const processLine = function* (line: string): Generator<StreamChunk | { __stopError?: boolean }> {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith(':')) return;
      if (!trimmed.startsWith('data:')) return;

      const data = trimmed.slice(5).trimStart();
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        return;
      }

      if (parsed.error) {
        yield {
          type: 'error',
          error: `Google API stream error: ${parsed.error.message || JSON.stringify(parsed.error)}`,
        };
        yield { __stopError: true };
        return;
      }

      // Process candidates
      const candidates = parsed.candidates;
      if (candidates && candidates.length > 0) {
        if (candidates[0].finishReason) {
          lastFinishReason = candidates[0].finishReason;
        }
        const content = candidates[0].content;
        if (content?.parts) {
          for (const part of content.parts) {
            if (part.text && !part.thought) {
              hasEmittedContent = true;
              yield { type: 'text', text: part.text };
            }
            if (part.functionCall) {
              hasEmittedContent = true;
              yield {
                type: 'tool_call',
                toolCall: {
                  id: `call_gemini_${nanoid(10)}`,
                  type: 'function',
                  function: {
                    name: part.functionCall.name,
                    arguments: JSON.stringify(part.functionCall.args || {}),
                  },
                  ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {}),
                },
              };
            }
          }
        }
      }

      // Usage metadata
      if (parsed.usageMetadata) {
        promptTokens = parsed.usageMetadata.promptTokenCount || 0;
        completionTokens = parsed.usageMetadata.candidatesTokenCount || 0;
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

    if (!hasEmittedContent && lastFinishReason && lastFinishReason !== 'STOP') {
      yield {
        type: 'error',
        error: `Google Gemini stopped without output (${lastFinishReason})`,
      };
      return;
    }

    yield { type: 'done', usage: { promptTokens, completionTokens } };
  }

  async listModels(): Promise<Model[]> {
    if (!this.apiKey) return [];

    try {
      const response = await fetch(`${this.baseUrl}/models?key=${this.apiKey}`);
      if (!response.ok) return [];

      const data = await response.json() as {
        models?: {
          name: string;
          displayName: string;
          inputTokenLimit?: number;
          supportedGenerationMethods?: string[];
        }[];
      };

      return (data.models || [])
        .filter((m) => m.supportedGenerationMethods?.includes('generateContent'))
        .map((m) => {
          const id = m.name.replace('models/', '');
          return {
            id,
            name: m.displayName || id,
            contextLength: m.inputTokenLimit,
            supportsTools: true,
          };
        });
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
