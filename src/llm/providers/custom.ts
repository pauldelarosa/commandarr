import type { LLMProvider, Model, Message, ToolDef, StreamChunk, ChatOptions } from '../provider';
import { formatOpenAIMessages, streamOpenAICompatibleChat } from './openai';

export class CustomProvider implements LLMProvider {
  id = 'custom';
  name = 'Custom (OpenAI-compatible)';

  private baseUrl = '';
  private apiKey = '';
  private model = '';

  configSchema: LLMProvider['configSchema'] = {
    fields: [
      {
        key: 'baseUrl',
        label: 'Base URL',
        type: 'url',
        required: true,
        placeholder: 'https://your-api.example.com/v1',
        helpText: 'Base URL of your OpenAI-compatible API (include /v1 if needed)',
      },
      {
        key: 'apiKey',
        label: 'API Key',
        type: 'password',
        required: false,
        placeholder: 'sk-...',
        helpText: 'API key for authentication (leave blank if not required)',
      },
      {
        key: 'model',
        label: 'Model',
        type: 'text',
        required: true,
        placeholder: 'model-name',
        helpText: 'Model identifier to use',
      },
    ],
    testPrompt: 'Say "hello" in one word.',
  };

  configure(config: Record<string, string>): void {
    this.baseUrl = (config.baseUrl || '').replace(/\/+$/, '');
    this.apiKey = config.apiKey || '';
    this.model = config.model || '';
  }

  async *chat(messages: Message[], tools?: ToolDef[], options?: ChatOptions): AsyncGenerator<StreamChunk> {
    if (!this.baseUrl) {
      yield { type: 'error', error: 'Custom provider base URL is not configured' };
      return;
    }
    if (!this.model) {
      yield { type: 'error', error: 'Custom provider model is not configured' };
      return;
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

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
        signal: AbortSignal.timeout(120_000),
      });
    } catch (err: any) {
      yield { type: 'error', error: `Custom provider request failed: ${err.message}` };
      return;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      yield { type: 'error', error: `Custom provider API error ${response.status}: ${errorText}` };
      return;
    }

    yield* streamOpenAICompatibleChat(response, 'Custom provider');
  }

  async listModels(): Promise<Model[]> {
    if (!this.baseUrl) return [];

    const headers: Record<string, string> = {};
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    try {
      const response = await fetch(`${this.baseUrl}/models`, { headers });
      if (!response.ok) return [];

      const data = await response.json() as { data?: { id: string }[] };
      return (data.data || []).map((m) => ({
        id: m.id,
        name: m.id,
        supportsTools: true,
      }));
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
