/**
 * ai-sdk LanguageModelV2 implementation that dispatches via the `claude --print`
 * CLI subprocess. Used by the `claude-cli` recipe to route gateway.toolLoop /
 * gateway.chat calls through Claude Code's OAuth session instead of the
 * Anthropic SDK + ANTHROPIC_API_KEY.
 *
 * Per-call routing is the contract: the gateway resolves the model string
 * to this recipe based on the `claude-cli:` prefix, instantiates one of
 * these objects per modelId, and dispatches doGenerate. Sibling subagent
 * jobs with `litellm:gpt-5.4` continue routing through litellm-proxy in
 * the same worker; no env-var switch, no global state.
 *
 * Tool use is supported via system-prompt-instructed JSON emission:
 *   The recipe injects a fenced instruction block into the system prompt
 *   that teaches the model the `<use_tools>[{id,name,input}, ...]</use_tools>`
 *   emission format. The adapter parses those blocks back into ai-sdk
 *   `tool-call` content parts. Parallel tool calls (multiple entries in
 *   the JSON array) round-trip cleanly — this is the case that breaks
 *   on the codex-proxy / litellm GPT-5.x bridge today. The protocol
 *   helpers live in cli-tool-protocol.ts, shared with
 *   codex-cli-language-model.ts; only the transport is claude-specific here.
 *
 * Context isolation:
 *   The subprocess is spawned from a dedicated tmpdir so claude-cli's
 *   CLAUDE.md auto-discovery has no local files to find. `--system-prompt`
 *   replaces the default system prompt; `--disable-slash-commands` skips
 *   skill resolution. User-level ~/.claude/CLAUDE.md still loads because
 *   the only way to skip it is `--bare`, which forces ANTHROPIC_API_KEY
 *   auth and defeats the whole point of this provider. The ~42k cached
 *   tokens from user-level instructions are accepted as a cost-trivial
 *   trade-off on the subscription path.
 *
 * doStream is not yet implemented; the model declares no streaming. Callers
 * (gateway.toolLoop primarily) use doGenerate.
 */
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2Content,
} from '@ai-sdk/provider';
import {
  buildToolUseInstructions,
  extractToolCalls,
  normalizeModel,
  renderPrompt,
} from './cli-tool-protocol.ts';

function claudeBin(): string {
  return process.env.GBRAIN_CLAUDE_CLI_BIN ?? 'claude';
}
const CLAUDE_CWD = join(tmpdir(), `gbrain-claude-cli-cwd-${process.pid}`);
let cwdEnsured = false;
function ensureCleanCwd(): string {
  if (!cwdEnsured) {
    mkdirSync(CLAUDE_CWD, { recursive: true });
    cwdEnsured = true;
  }
  return CLAUDE_CWD;
}

/** Parsed shape of `claude --print --output-format json`. */
interface ClaudeJsonResult {
  type: 'result';
  subtype: 'success' | string;
  is_error: boolean;
  result: string;
  stop_reason: string | null;
  session_id: string;
  num_turns: number;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
}

/**
 * Spawn `claude --print` with the contamination-suppression flags and return
 * the parsed `--output-format json` envelope. Aborts propagate to SIGTERM on
 * the child.
 */
function runClaude(
  systemPrompt: string,
  userPrompt: string,
  model: string,
  signal?: AbortSignal,
): Promise<ClaudeJsonResult> {
  return new Promise((resolve, reject) => {
    const args = [
      '--print',
      '--output-format', 'json',
      '--model', model,
      '--disable-slash-commands',
      // Agent isolation: this subprocess must behave like a raw LLM, not a
      // full Claude Code agent. `--tools ""` disables every built-in tool
      // (Bash/Read/WebSearch/...); `--strict-mcp-config` ignores all user-level
      // MCP servers (without it, each call would boot the user's MCP servers —
      // including gbrain's own MCP → recursion + PGLite single-writer lock
      // contention). Verified against claude CLI 2.1.145 --help.
      '--tools', '',
      '--strict-mcp-config',
    ];
    if (systemPrompt) {
      args.push('--system-prompt', systemPrompt);
    }
    // Env scrub: guarantee the CLI authenticates via its own OAuth session
    // (subscription), never via an inherited API key. Without this, an
    // ANTHROPIC_API_KEY in gbrain's env (the exact setup this recipe is meant
    // to replace) silently flips billing to per-token API usage.
    const env = { ...process.env };
    delete env.ANTHROPIC_API_KEY;
    delete env.ANTHROPIC_AUTH_TOKEN;
    delete env.ANTHROPIC_BASE_URL;
    const child = spawn(claudeBin(), args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: ensureCleanCwd(),
      env,
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += String(chunk); });
    child.stderr.on('data', chunk => { stderr += String(chunk); });

    const onAbort = () => {
      child.kill('SIGTERM');
      reject(new Error('claude-cli adapter aborted'));
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    child.on('error', err => {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`claude-cli spawn failed: ${err instanceof Error ? err.message : String(err)}`));
    });

    child.on('close', code => {
      if (signal) signal.removeEventListener('abort', onAbort);
      if (code !== 0) {
        reject(new Error(`claude-cli exited ${code}: ${stderr.trim() || stdout.trim()}`));
        return;
      }
      try {
        let parsed = JSON.parse(stdout) as unknown;
        // Compat: when the user has `"verbose": true` in ~/.claude/settings.json,
        // `--print --output-format json` emits an ARRAY of events
        // ([{type:"system",subtype:"init",...}, ..., {type:"result",...}])
        // instead of the bare result object. There is no CLI flag to force it
        // off (no --no-verbose; --settings '{}' merges, does not replace), so
        // tolerate both shapes and pick the result event. Verified on CLI 2.1.145.
        if (Array.isArray(parsed)) {
          const resultEvent = parsed.find(
            (ev): ev is ClaudeJsonResult =>
              !!ev && typeof ev === 'object' && (ev as { type?: unknown }).type === 'result',
          );
          if (!resultEvent) {
            reject(new Error(`claude-cli JSON event array had no "result" event\n--- raw ---\n${stdout.slice(0, 500)}`));
            return;
          }
          parsed = resultEvent;
        }
        const envelope = parsed as ClaudeJsonResult;
        if (envelope.is_error) {
          reject(new Error(`claude-cli reported error: ${envelope.result || envelope.subtype}`));
          return;
        }
        resolve(envelope);
      } catch (e) {
        reject(new Error(`claude-cli output not JSON: ${e instanceof Error ? e.message : String(e)}\n--- raw ---\n${stdout.slice(0, 500)}`));
      }
    });

    // stdin error handler: if the binary does not exist (ENOENT) or the child
    // dies before draining stdin, write/end can emit an unhandled 'error'
    // (EPIPE) that would crash the worker. The spawn-level 'error' / non-zero
    // 'close' handlers above already surface the real failure, so the stdin
    // error itself is safe to swallow.
    child.stdin.on('error', () => { /* surfaced via child 'error'/'close' */ });
    try {
      child.stdin.write(userPrompt);
      child.stdin.end();
    } catch (e) {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`claude-cli stdin write failed (is the claude binary installed?): ${e instanceof Error ? e.message : String(e)}`));
    }
  });
}

export class ClaudeCliLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly provider = 'claude-cli';
  readonly modelId: string;
  readonly supportedUrls = {};

  constructor(modelId: string) {
    this.modelId = normalizeModel(modelId);
  }

  async doGenerate(options: LanguageModelV2CallOptions): Promise<{
    content: LanguageModelV2Content[];
    finishReason: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';
    usage: { inputTokens: number | undefined; outputTokens: number | undefined; totalTokens: number | undefined };
    warnings: never[];
  }> {
    const { systemText, userPrompt } = renderPrompt(options.prompt);
    const toolInstructions = buildToolUseInstructions(options.tools);
    const systemPrompt = [systemText, toolInstructions].filter(s => s.length > 0).join('\n');

    const result = await runClaude(systemPrompt, userPrompt, this.modelId, options.abortSignal);
    const { toolCalls, beforeText, afterText } = extractToolCalls(result.result, 'toolu_claude_cli_');

    const content: LanguageModelV2Content[] = [];
    if (beforeText) content.push({ type: 'text', text: beforeText });
    for (const call of toolCalls) {
      content.push({
        type: 'tool-call',
        toolCallId: call.id,
        toolName: call.name,
        input: call.input,
      });
    }
    if (afterText) content.push({ type: 'text', text: afterText });
    if (content.length === 0) {
      // Empty response — still hand the caller a well-formed content array.
      content.push({ type: 'text', text: result.result ?? '' });
    }

    const finishReason = toolCalls.length > 0 ? 'tool-calls' as const : 'stop' as const;
    const inputTokens = result.usage?.input_tokens;
    const outputTokens = result.usage?.output_tokens;
    const totalTokens = (inputTokens ?? 0) + (outputTokens ?? 0);

    return {
      content,
      finishReason,
      usage: {
        inputTokens,
        outputTokens,
        totalTokens: inputTokens !== undefined && outputTokens !== undefined ? totalTokens : undefined,
      },
      warnings: [],
    };
  }

  async doStream(): Promise<never> {
    throw new Error(
      'claude-cli LanguageModel does not support streaming. Use doGenerate or set ' +
      'the model on a non-streaming chat surface (gateway.toolLoop is non-streaming).',
    );
  }
}
