/**
 * ai-sdk LanguageModelV2 implementation that dispatches via the `codex exec`
 * CLI subprocess. Used by the `codex-cli` recipe to route gateway.toolLoop /
 * gateway.chat calls through the Codex CLI's ChatGPT OAuth session instead of
 * the OpenAI SDK + OPENAI_API_KEY.
 *
 * Per-call routing is the contract: the gateway resolves the model string
 * to this recipe based on the `codex-cli:` prefix, instantiates one of
 * these objects per modelId, and dispatches doGenerate. Sibling subagent
 * jobs with `claude-cli:...` or `litellm:...` continue routing through their
 * own providers in the same worker; no env-var switch, no global state.
 *
 * Tool use is supported via prompt-instructed JSON emission — the SAME
 * `<use_tools>[{id,name,input}, ...]</use_tools>` protocol the claude-cli
 * provider teaches. The protocol helpers live in cli-tool-protocol.ts,
 * shared with claude-cli-language-model.ts; only the transport (spawn
 * flags, env scrubbing, output channel) is codex-specific here.
 *
 * Context isolation:
 *   - Spawned from a dedicated tmpdir (`-C`) so AGENTS.md auto-discovery has
 *     no local files to find; `--skip-git-repo-check` skips the repo probe.
 *     Sandbox-confined codex builds (notably the Ubuntu snap) cannot open
 *     arbitrary paths under /tmp, which surfaces as a bare "No such file or
 *     directory" from the child; set TMPDIR to a non-hidden directory inside
 *     $HOME on those hosts. See docs/ai-providers/codex-cli.md.
 *   - `--ignore-user-config` stops ~/.codex/config.toml from loading — user
 *     MCP servers (including gbrain's own MCP → recursion + PGLite lock
 *     contention), custom model defaults, and instruction overrides all stay
 *     out of the subprocess. Auth state (auth.json) still loads.
 *   - `--sandbox read-only` pins the agent sandbox down for defense in
 *     depth; with the tool-use protocol the model answers in text and has
 *     nothing to execute anyway.
 *   - Codex has no `--system-prompt` flag, so system messages are rendered
 *     as a leading `## System` section of the stdin prompt.
 *
 * Output channel: `-o <file>` writes the agent's final message verbatim;
 * stdout carries progress logs and is discarded. Token usage is not exposed
 * on this channel, so usage fields are undefined (the budget ledger treats
 * subscription-billed calls as nominal anyway — see the recipe comment).
 *
 * doStream is not yet implemented; the model declares no streaming. Callers
 * (gateway.toolLoop primarily) use doGenerate.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync } from 'node:fs';
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

function codexBin(): string {
  return process.env.GBRAIN_CODEX_CLI_BIN ?? 'codex';
}
const CODEX_CWD = join(tmpdir(), `gbrain-codex-cli-cwd-${process.pid}`);
let cwdEnsured = false;
function ensureCleanCwd(): string {
  if (!cwdEnsured) {
    mkdirSync(CODEX_CWD, { recursive: true });
    cwdEnsured = true;
  }
  return CODEX_CWD;
}

/**
 * Explain a child failure that is really a cwd-reachability problem.
 *
 * The dedicated cwd is handed to both `-C` and `-o`, so the child must be able
 * to open it. Sandbox-confined builds cannot: snap's `home` interface allows
 * only non-hidden paths under $HOME, so nothing under /tmp is reachable and
 * codex reports a bare "No such file or directory (os error 2)" that names
 * neither the path nor the cause. Point at the cwd and the TMPDIR escape hatch
 * instead of leaving the operator with the raw errno.
 *
 * Returns '' for unrelated failures so ordinary CLI errors stay untouched.
 */
function cwdAccessHint(detail: string): string {
  if (!/no such file or directory|os error 2|permission denied|os error 13/i.test(detail)) return '';
  return (
    `\n--- hint ---\ncodex ran with cwd ${CODEX_CWD}. Sandbox-confined installs ` +
    `(e.g. the Ubuntu snap) cannot open arbitrary paths under ${tmpdir()}; set ` +
    `TMPDIR to a non-hidden directory inside $HOME and retry.`
  );
}

/**
 * Spawn `codex exec` with the contamination-suppression flags and return the
 * final agent message from the `-o` output file. Aborts propagate to SIGTERM
 * on the child.
 */
function runCodex(
  fullPrompt: string,
  model: string,
  signal?: AbortSignal,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const outFile = join(
      ensureCleanCwd(),
      `codex-out-${Date.now()}-${Math.random().toString(36).slice(2, 10)}.txt`,
    );
    const args = [
      'exec',
      // Agent isolation: this subprocess must behave like a raw LLM, not a
      // full Codex agent. `--ignore-user-config` stops ~/.codex/config.toml
      // (user MCP servers, model defaults, instruction overrides) from
      // loading — without it, each call would boot the user's MCP servers,
      // including gbrain's own MCP → recursion + PGLite single-writer lock
      // contention. `--sandbox read-only` pins the sandbox for defense in
      // depth. `-C` + `--skip-git-repo-check` keep AGENTS.md discovery and
      // the repo probe out of a clean tmpdir.
      '--ignore-user-config',
      '--sandbox', 'read-only',
      '--skip-git-repo-check',
      '-C', ensureCleanCwd(),
      '-m', model,
      '-o', outFile,
      // Read the prompt from stdin — argv has a hard size ceiling and
      // subagent prompts (context + tool specs) routinely exceed it.
      '-',
    ];
    // Env scrub: guarantee the CLI authenticates via its own OAuth session
    // (subscription), never via an inherited API key. Without this, an
    // OPENAI_API_KEY in gbrain's env (the exact setup this recipe is meant
    // to replace) silently flips billing to per-token API usage.
    const env = { ...process.env };
    delete env.OPENAI_API_KEY;
    delete env.OPENAI_BASE_URL;
    const child = spawn(codexBin(), args, {
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
      reject(new Error('codex-cli adapter aborted'));
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    const readOutFile = (): string | null => {
      try {
        const text = readFileSync(outFile, 'utf8');
        rmSync(outFile, { force: true });
        return text;
      } catch {
        return null;
      }
    };

    child.on('error', err => {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`codex-cli spawn failed: ${err instanceof Error ? err.message : String(err)}`));
    });

    child.on('close', code => {
      if (signal) signal.removeEventListener('abort', onAbort);
      if (code !== 0) {
        const detail = stderr.trim() || stdout.trim();
        reject(new Error(`codex-cli exited ${code}: ${detail}${cwdAccessHint(detail)}`));
        return;
      }
      const text = readOutFile();
      if (text === null || text.trim().length === 0) {
        reject(new Error(
          `codex-cli exited 0 but wrote no final message to -o file\n--- stderr ---\n${stderr.slice(0, 500)}`,
        ));
        return;
      }
      resolve(text.trim());
    });

    // stdin error handler: if the binary does not exist (ENOENT) or the child
    // dies before draining stdin, write/end can emit an unhandled 'error'
    // (EPIPE) that would crash the worker. The spawn-level 'error' / non-zero
    // 'close' handlers above already surface the real failure, so the stdin
    // error itself is safe to swallow.
    child.stdin.on('error', () => { /* surfaced via child 'error'/'close' */ });
    try {
      child.stdin.write(fullPrompt);
      child.stdin.end();
    } catch (e) {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`codex-cli stdin write failed (is the codex binary installed?): ${e instanceof Error ? e.message : String(e)}`));
    }
  });
}

export class CodexCliLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly provider = 'codex-cli';
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
    // No --system-prompt flag on codex exec: system text + tool protocol
    // lead the stdin prompt as a `## System` section instead.
    const fullPrompt = [
      systemText ? `## System\n\n${systemText}` : '',
      toolInstructions,
      userPrompt,
    ].filter(s => s.length > 0).join('\n\n');

    const raw = await runCodex(fullPrompt, this.modelId, options.abortSignal);
    const { toolCalls, beforeText, afterText } = extractToolCalls(raw, 'toolu_codex_cli_');

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
      content.push({ type: 'text', text: raw });
    }

    const finishReason = toolCalls.length > 0 ? 'tool-calls' as const : 'stop' as const;

    return {
      content,
      finishReason,
      // The -o output channel carries no token accounting; leave usage
      // undefined rather than fabricate numbers (subscription billing makes
      // the ledger nominal for this recipe anyway).
      usage: { inputTokens: undefined, outputTokens: undefined, totalTokens: undefined },
      warnings: [],
    };
  }

  async doStream(): Promise<never> {
    throw new Error(
      'codex-cli LanguageModel does not support streaming. Use doGenerate or set ' +
      'the model on a non-streaming chat surface (gateway.toolLoop is non-streaming).',
    );
  }
}
