import type { LLMProvider, Model, Message, ToolDef, StreamChunk, ChatOptions } from '../provider';
import { formatOpenAIMessages, streamOpenAICompatibleChat } from './openai';

export class LMStudioProvider implements LLMProvider {
  id = 'lmstudio';
  name = 'LM Studio';

  private baseUrl = 'http://localhost:1234';
  private model = '';

  configSchema: LLMProvider['configSchema'] = {
    fields: [
      {
        key: 'baseUrl',
        label: 'Base URL',
        type: 'url',
        required: false,
        default: 'http://localhost:1234',
        placeholder: 'http://localhost:1234',
        helpText: 'LM Studio server address',
      },
      {
        key: 'model',
        label: 'Model',
        type: 'text',
        required: true,
        placeholder: 'loaded model name',
        helpText: 'Model identifier loaded in LM Studio. Leave empty to use the currently loaded model.',
      },
    ],
    testPrompt: 'Say "hello" in one word.',
  };

  configure(config: Record<string, string>): void {
    this.baseUrl = (config.baseUrl || 'http://localhost:1234').replace(/\/+$/, '');
    this.model = config.model || '';
  }

  async *chat(messages: Message[], tools?: ToolDef[], options?: ChatOptions): AsyncGenerator<StreamChunk> {
    const body: Record<string, unknown> = {
      model: this.model || 'default',
      max_tokens: options?.maxTokens ?? 4096,
      messages: formatOpenAIMessages(messages),
      stream: true,
    };
    if (options?.temperature !== undefined) body.temperature = options.temperature;
    if (tools && tools.length > 0) body.tools = tools;

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(120_000),
      });
    } catch (err: any) {
      yield {
        type: 'error',
        error: `LM Studio connection failed: ${err.message}. Is LM Studio running at ${this.baseUrl}?`,
      };
      return;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      yield { type: 'error', error: `LM Studio API error ${response.status}: ${errorText}` };
      return;
    }

    yield* streamOpenAICompatibleChat(response, 'LM Studio');
  }

  async listModels(): Promise<Model[]> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/models`);
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
    // LM Studio tool support depends on the loaded model
    return true;
  }

  supportsStreaming(): boolean {
    return true;
  }
}
