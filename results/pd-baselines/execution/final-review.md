# Independent whole-change review and disposition

Reviewer: fresh gpt-6-astra context, one continuous read-only review, no subagents.
Range: 2882e39163b3eecd4add60633a0f86a1f6c466a5..fa28ab5a2854db44a717a7a88c5796020bda2eb6.
The fixed-code pass overlapped the final repetitions; the verdict followed the
final ready marker, committed canonical data, and restored-service evidence.

The reviewer independently verified all590 raw-source hashes, byte-identical
double exports,45 distinct formal server identities,1110 strict tool-valid
responses,30 full-length long completions,zero formal errors,actual SGLang C4
and152639-token capacity,live restored PID3433597/exact argv/two manifests,
recorded OpenAI/Anthropic tool checks,and GPU0's original process.

Critical: none. Original verdict: ready with fixes.

Important findings:

1. SIGTERM/SIGHUP did not unwind the PD controller's restoration context.
2. Waiting only for the frontend allowed TERM-ignoring workers to survive,
   including successful external-command completion, and block restoration.

Minor finding: the canonical exporter omitted recorded48s completion counts.
The executor regraded this Important because standalone canonical results
otherwise cannot support the promised fixed-window capacity analysis.

One consolidated fix pass:

- Scoped controller handlers unwind on the first interruption, protect restore
  from subsequent interruptions,and restore prior handlers. Spawn/cleanup
  critical sections defer signals until ownership is recorded or cleanup ends.
- Private groups keep an unreaped leader anchor through TERM/deadline/KILL;
  successful external commands also clean descendants. The live identity
  mismatch refusal is preserved.
- The exporter recomputes48s protocol and strict-tool counts separately,
  including the exact boundary,and leaves missing timing unknown.

Actual RED:9 initial regressions failed;6 signal variants failed the old
controller;4 additional spawn/shutdown interruption variants exposed surviving
workers before critical-section deferral. Actual GREEN:15 new regressions
passed7.32s; full project161 passed23.21s. See adjacent raw logs and task4 log.
The v2 derived exports are byte-identical twice; all previously exported group
fields and all raw source hashes are unchanged. No second review was dispatched.

Declined to judge, exhaustively, with executor rulings in progress.md:

- Upstream MorphX efficacy and GPU efficacy of PDMux/literature-only methods:
  local compatibility evidence cannot establish those performance results.
- Broad quality equivalence: code/tool gates do not establish general accuracy.
- Isolated scheduler causality/global novelty/optimal ceilings: precision,
  capacity,residual-input confounds and unmeasured methods remain.
- SIGKILL/host/power failure: an external supervisor is needed beyond catchable
  in-process interruption recovery.

Deferred minors: none; the sole Minor was regraded and fixed.
