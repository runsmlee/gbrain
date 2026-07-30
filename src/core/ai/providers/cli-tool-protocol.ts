/**
 * Shared prompt-instructed tool-call protocol for CLI subprocess providers.
 *
 * claude-cli and codex-cli both drive a coding-agent CLI as a raw LLM over
 * stdin/stdout. Neither transport has native tool-calling, so both teach the
 * model the same `<use_tools>[{id,name,input}, ...]</use_tools>` emission
 * format and parse it back into LanguageModelV2 tool-call parts. These
 * helpers are that protocol — everything transport-specific (spawn flags,
 * env scrubbing, output channels) stays in the per-provider modules.
 *
 * Extracted verbatim from claude-cli-language-model.ts once codex-cli became
 * the second consumer and proved the shape (see PR #3587's review notes).
 */
import type {
  LanguageModelV2FunctionTool,
  LanguageModelV2Prompt,
  LanguageModelV2Message,
  LanguageModelV2ProviderDefinedTool,
} from '@ai-sdk/provider';

/**
 * Build the system-prompt addendum that teaches the model the
 * `<use_tools>...</use_tools>` emission format. Returns the empty string
 * when no tools are registered for this turn so the model gets a normal
 * text-completion prompt without protocol noise.
 */
export function buildToolUseInstructions(
  tools: ReadonlyArray<LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool> | undefined,
): string {
  if (!tools || tools.length === 0) return '';

  const functionTools = tools.filter((t): t is LanguageModelV2FunctionTool => t.type === 'function');
  if (functionTools.length === 0) return '';

  const toolSpecs = functionTools.map(t => ({
    name: t.name,
    description: t.description ?? '',
    input_schema: t.inputSchema ?? { type: 'object', properties: {} },
  }));

  return [
    '',
    '## Tool Use Protocol',
    '',
    'You have access to these tools:',
    '',
    '```json',
    JSON.stringify(toolSpecs, null, 2),
    '```',
    '',
    'To call one or more tools in this turn, emit EXACTLY ONE block of this form, ' +
      'with no other text outside the block on its own lines:',
    '',
    '<use_tools>',
    '[',
    '  {"id": "<unique tool call id, like toolu_01ABC>", "name": "<tool name>", "input": <input object matching the tool\'s input_schema>}',
    ']',
    '</use_tools>',
    '',
    'Multiple tool calls go in the array. Tool results are returned to you on the ' +
      'next turn as [tool_result <text>] entries. You may then call more tools or emit a final response.',
    '',
    'When you are ready to give a final answer instead of calling tools, respond with prose text only — ' +
      'do not include a <use_tools> block in that case.',
    '',
  ].join('\n');
}

/**
 * Render the ai-sdk message array into a single text prompt for the CLI's
 * stdin. System messages are extracted separately — how they reach the CLI
 * is the caller's transport decision (claude-cli: `--system-prompt` flag;
 * codex-cli: leading `## System` stdin section). Tool calls and tool results
 * are rendered as placeholders so the model sees the conversation in a
 * coherent shape even though the adapter does not natively round-trip tool
 * calls through the CLI.
 */
export function renderPrompt(prompt: LanguageModelV2Prompt): { systemText: string; userPrompt: string } {
  const systemParts: string[] = [];
  const convo: string[] = [];

  for (const msg of prompt as ReadonlyArray<LanguageModelV2Message>) {
    if (msg.role === 'system') {
      systemParts.push(msg.content);
      continue;
    }
    if (msg.role === 'user') {
      const text = msg.content
        .map(p => {
          if (p.type === 'text') return p.text;
          // File parts get a stub — multimodal is not supported via subprocess yet.
          if (p.type === 'file') return `[file ${p.mediaType ?? 'unknown'}]`;
          return '';
        })
        .filter(s => s.length > 0)
        .join('\n');
      if (text) convo.push(`User: ${text}`);
      continue;
    }
    if (msg.role === 'assistant') {
      const rendered = msg.content
        .map(p => {
          if (p.type === 'text') return p.text;
          if (p.type === 'reasoning') return ''; // dropped on replay
          if (p.type === 'tool-call') {
            return `[tool_use ${p.toolName}(${p.input})]`;
          }
          if (p.type === 'tool-result') {
            const out = typeof p.output === 'string' ? p.output : JSON.stringify(p.output);
            return `[tool_result ${out}]`;
          }
          return '';
        })
        .filter(s => s.length > 0)
        .join('\n');
      if (rendered) convo.push(`Assistant: ${rendered}`);
      continue;
    }
    if (msg.role === 'tool') {
      const rendered = msg.content
        .map(p => {
          const out = typeof p.output === 'string' ? p.output : JSON.stringify(p.output);
          return `[tool_result ${out}]`;
        })
        .join('\n');
      if (rendered) convo.push(`User: ${rendered}`);
      continue;
    }
  }

  return { systemText: systemParts.join('\n'), userPrompt: convo.join('\n\n') };
}

export interface ParsedToolCall {
  id: string;
  name: string;
  /** Stringified JSON, matching the ai-sdk LanguageModelV2ToolCall.input contract. */
  input: string;
}

/**
 * Locate and parse the `<use_tools>...</use_tools>` block in the assistant's
 * raw text response. Returns the parsed tool calls plus whatever prose
 * surrounded the block. Returns an empty `toolCalls` array when no block is
 * present, malformed, or unterminated — the caller then treats the full
 * raw text as a final text response.
 *
 * `idPrefix` names the provider in ids synthesized for entries the model
 * emitted without one (e.g. `toolu_claude_cli_`), so a mixed-provider trace
 * still says which adapter minted the id.
 */
export function extractToolCalls(raw: string, idPrefix: string): {
  toolCalls: ParsedToolCall[];
  beforeText: string;
  afterText: string;
} {
  const openTag = '<use_tools>';
  const closeTag = '</use_tools>';
  const openIdx = raw.indexOf(openTag);
  if (openIdx === -1) {
    return { toolCalls: [], beforeText: raw.trim(), afterText: '' };
  }
  const closeIdx = raw.indexOf(closeTag, openIdx + openTag.length);
  if (closeIdx === -1) {
    // Unterminated block — recover gracefully.
    return { toolCalls: [], beforeText: raw.trim(), afterText: '' };
  }

  const beforeText = raw.slice(0, openIdx).trim();
  const afterText = raw.slice(closeIdx + closeTag.length).trim();
  let inner = raw.slice(openIdx + openTag.length, closeIdx).trim();

  if (inner.startsWith('```')) {
    inner = inner.replace(/^```(?:json|JSON)?\s*\n?/, '').replace(/\n?```$/, '').trim();
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(inner);
  } catch {
    return { toolCalls: [], beforeText: raw.trim(), afterText: '' };
  }
  if (!Array.isArray(parsed)) {
    return { toolCalls: [], beforeText: raw.trim(), afterText: '' };
  }

  const toolCalls: ParsedToolCall[] = [];
  for (const entry of parsed) {
    if (!entry || typeof entry !== 'object') continue;
    const e = entry as Record<string, unknown>;
    const name = typeof e.name === 'string' ? e.name : null;
    if (!name) continue;
    const id = typeof e.id === 'string' && e.id.length > 0
      ? e.id
      : `${idPrefix}${Math.random().toString(36).slice(2, 12)}`;
    const inputJson = JSON.stringify(e.input ?? {});
    toolCalls.push({ id, name, input: inputJson });
  }

  return { toolCalls, beforeText, afterText };
}

/**
 * Strip provider prefixes (`anthropic:`, `claude-cli:`, `codex-cli:`,
 * `litellm:`, ...) that the underlying CLI does not understand. The gateway
 * hands us a bare model id via `recipe.aliases` resolution, but defensive
 * normalization here keeps direct LanguageModelV2 construction (in tests,
 * for example) ergonomic.
 */
export function normalizeModel(model: string): string {
  const idx = model.indexOf(':');
  return idx >= 0 ? model.slice(idx + 1) : model;
}
