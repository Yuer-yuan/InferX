# Completed-step timing source audit

These five installed vLLM source files were copied read-only after the initial
profile. The archive SHA256 is3991abb9ec5a58d441c47e1795d13b89dad55e8bc11b315695247e058662b0fb.
Every copied file hash was checked. Recorded installed-file modification/change
timestamps precede the actual initial profile. This supplemental capture is
separate from the twelve source files guarded at scheduler import time.

For the measured synchronous MTP0/UniProc configuration:

1. Pinned EngineCore.step waits for execute_model's future, calls sample_tokens
   when needed, and only then calls native scheduler.update_from_output.
2. GPUModelRunner.sample_tokens calls _bookkeeping_sync. With async scheduling
   disabled, it calls _to_list on the sampled GPU tensor before clearing the
   discarded prefill samples.
3. _to_list copies the tensor to pinned CPU storage, records transfer_event and
   synchronizes that event before returning the CPU list. Thus even discarded
   prefill outputs pass through the existing completion boundary.
4. UniProcExecutor.sample_tokens uses the blocking collective_rpc path. Its
   separate AsyncOutputFuture path also resolves get_output, but that is not
   the synchronous path relied upon here.

No extra CUDA synchronization was added by this audit or the collector.
The code establishes the model-output completion ordering; it does not identify
why fixed and cycling requests with identical recorded features differ in time.
The matched-feature empirical audit is ../independent-profile-v5/matched-shape-audit.json.
