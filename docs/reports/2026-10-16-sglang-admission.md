# SGLang 0.5.19 admission diagnosis — local evidence only

The strongest concrete explanation is a request-row accounting interaction on the final chunk of a prefill: the continuing request already owns a pool row, but is counted again in `len(adder.can_run_list)` when comparing against the number of **free** rows. With two decoders plus a continuing third request and a fourth waiting, the comparison becomes `1 >= min(4 - 2, 1)`. It sets `running_batch.batch_is_full=True`. After the third request merges into decode, that flag suppresses new prefills until a decoder finishes. This source path predicts three decoders plus one queued at C4, and three plus five queued at C8.

This is a source-derived causal candidate with unusually close agreement to the saved runtime evidence. The branch operands and flag transitions were not logged, so the saved artifacts do not establish that this exact branch fired. Do not call it an instrumented, experimentally confirmed root cause, or attribute the result to Mamba memory exhaustion.

## Scope and provenance

Read-only local source/config/client/log inspection; no remote commands, API calls, GPU actions, configuration/source changes, or subagents. The diagnostic report was produced without changing the running services. The ongoing performance/quality campaign is unchanged. Applied the repository's systematic-debugging investigation workflow.

The wheel SHA256 was recomputed as `fdc65d1235f6861ad3a1116c36c2270e46a95f030200091f8de595c6c3426cd2`. All eight source files below were compared directly with their corresponding ZIP member bytes and matched. All source references below are under `artifacts/framework-c4/sglang/wheel/sglang/srt/`:

| File | SHA256 |
| --- | --- |
| `managers/scheduler.py` | `3a12fdfb9c21571d8007ba60aedd7e84b67636583bac8243373c75d6a2ab324e` |
| `managers/schedule_policy.py` | `5ceae6ad98493297bf557fdb38b6e53f654319e26238c9e4e22a00aead230e3c` |
| `managers/schedule_batch.py` | `170f847aba6701d8d876a98c5edb8a66f188f5bc12fc30973596d132fcbb8c72` |
| `managers/scheduler_components/batch_result_processor.py` | `0f0a3d92603c58bc555969b4f01daf2796f5aac9f38ad5a7155455ef5ca3ed4a` |
| `mem_cache/memory_pool.py` | `7955db1470ea36b99eb43483a5bd999dbda62119ff54aee7f9e8e01f31b2a9b5` |
| `mem_cache/mamba_radix_cache.py` | `464250e75cf4e4d4cb2deedf862c188cf231b9f9f8b250de04d10a00b702f1ce` |
| `mem_cache/common.py` | `a209daeb493fc701e56e452dfcd336520ae8a7343f76faa3cd6d1b956039ea52` |
| `mem_cache/kv_cache_configurator.py` | `40a4d40ba17dbd5065048a96077484b3158a26c00638a087c4678569597f5f04` |

## What the source proves

1. **Four is the resolved limit, not three.** `official-native-32k/command.json` explicitly passes `--max-running-requests 4`. The resolved internal state in `official-native-32k/server-info.json` and C4 `server-info-before.json` has `max_running_requests=4`, `effective_max_running_requests_per_dp=4`, `pp_max_micro_batch_size=4`, TP1/PP1, chunk4096, mixed_chunk false, overlap enabled, radix enabled, `max_mamba_cache_size=44`, unified memory false. `kv_cache_configurator.py:2112` derives the request count from the user limit; `:515` supplies that count to the request pool, `:956` selects the hybrid pool, and `:1056` constructs it with `size=max_num_reqs`. `memory_pool.py:272` and `:282` create exactly `size` allocatable rows; index 0 is padding, not a fifth usable request.

2. **A continuing chunk keeps its request row.** `memory_pool.py:292` explicitly reuses rows for requests with `holds_kv`. `scheduler.py:3245` stashes a chunk through `maybe_cache_unfinished_req`; `common.py:161` only delegates to the tree cache. `mamba_radix_cache.py:674` through `:803` update cached KV, prefix indices, Mamba state and locks without freeing the request row. The only `free(self, req)` implementation in `memory_pool.py:333` clears that row; normal completed-request cleanup reaches it through `common.py:295`. Thus the nearby scheduler comment at `:3571` about chunk space having just been released does not establish row release in this implementation.

3. **The final chunk is counted before the waiting-request admission check.** `scheduler.py:3627` calls `adder.add_chunked_req` first. `schedule_policy.py:1005` computes truncation, `:1008` appends the continuing request to `can_run_list`, and `:1022` returns `None` when this is the last chunk, clearing `self.chunked_req`. At this point the request remains in `can_run_list` and retains its row.

4. **The waiting-request loop can mark the batch full while a row is free.** `scheduler.py:3484` returns `min(pp_max_micro_batch_size - running_bs, req_to_token_pool.available_size())`, with beam reservations subtracted; ordinary non-beam requests do not consume beam reservations. At `:3656`, `len(adder.can_run_list) >= get_num_allocatable_reqs(...)` sets `batch_is_full=True`, then `:3668` breaks admission. For two running decoders, one continuing third request and one waiting fourth request, a four-row pool has one free row. The continuing request contributes one to `can_run_list` even though its row was already deducted from the free-row count: `1 >= min(2,1)` is true. This does not require token or Mamba exhaustion.

5. **The flag can persist after decode reaches three.** The final chunk batch is constructed with `chunked_req=None` at `scheduler.py:3765`. On the next iteration it is merged into running decode at `:3387`–`:3412`; since it is neither excluded as an intermediate chunk nor finished, filtering need not shrink the batch. `schedule_batch.py:3463` merges requests without resetting `batch_is_full`. Then `scheduler.py:3551` skips prefill whenever this flag is set and no chunk is in progress. A decoder finishing and being removed clears the flag in `update_running_batch` at `:3873`/`:3943`; the empty-batch case also clears it. The result is a plausible wave of three requests even though the configured ceiling is four.

The queue being nonempty is material: with C3, the final chunk of the third request may have no waiting fourth request, so the loop containing `:3656` does not execute. With C4/C8, there is a waiting request, so this particular premature full flag has an opportunity to latch. The chunk budget also naturally limits these cold 8192-token prompts to one 4096-token prefill sequence per forward (`schedule_policy.py:1385`, `:810`, `:828`); exhausting this chunk budget returns `OTHER`, whereas the scheduler only treats `NO_TOKEN` as a token-pressure full flag at `scheduler.py:3714`. One sequence per prefill therefore does not itself prove memory pressure.

## What runtime artifacts establish

All log line references below mean the line number after decompressing the named `server-interval.log.gz`. These are bounded row intervals including warmup; decode messages occur every 40 steps and are not a complete forward trace.

Both mean and p95 latency are shown below; the headline comparisons use p95:

| Row | Mean TTFT | p95 TTFT | p95 TPOT / exported inter-token latency | Output throughput |
| --- | ---: | ---: | ---: | ---: |
| `05-c3-n24-s46` | 4.244 s | 5.686 s | 47.658 ms | 58.182 tok/s |
| `03-c4-n32-s44` | 12.500 s | 30.075 s | 47.669 ms | 56.829 tok/s |
| `04-c8-n24-s45` | 40.834 s | 58.481 s | 47.659 ms | 58.181 tok/s |

Source: each row's `summary.json`. C4's median TTFT is still 5.638 s; the severe C4 failure is the tail, while C8's median is 43.437 s.

- **C4:** `results/framework-c4/sglang/official-performance/03-c4-n32-s44/server-interval.log.gz:41`–`:48` show six 4096-token prefill messages with running counts 0,0,1,1,2,2; at the last one a request is still queued. Lines `:50`–`:62` then show running3/queued1 for approximately 19 seconds without another prefill message. Prefill resumes at `:63` with running0. The first warmup wave at `:4`–`:25` shows the same shape. The row audit counts 140 sampled decode messages at running3 and the same 140 at queue1, with maximum reported Mamba usage .27.
- **C8:** `.../04-c8-n24-s45/server-interval.log.gz:4`–`:11` show six chunks producing three requests; `:13`–`:25` show running3/queued5 until prefill resumes at `:26`. The row audit has 129 sampled running3 messages, including 90 queue5 messages, maximum Mamba usage .27.
- **C3:** `.../05-c3-n24-s46/server-interval.log.gz:4`–`:11` show the same six chunks, but no waiting request by the third request's final chunk; `:13`–`:25` show running3/queue0. All 128 sampled decode messages in the row have queue0.
- **Client timing corroborates queueing waves:** in C4 `profile_export.jsonl`, profiling records at lines 5–8 start within 2 ms: three have TTFT about 5.64–5.68 s and finish near 26.39 s, while the fourth has TTFT 28.424 s and finishes at 52.786 s, relative to the first profiling request's start. C8 lines 9–16 contain eight profiling starts within 2 ms, but the last two have TTFT 58.476 s. Consequently this is not merely a misleading low batch count sampled at arbitrary instants; actual clients wait for later waves.

## Alternatives and limits

The resolved four-request limit rules out an explicit configured cap of three. Four request rows and 44 Mamba state slots are different capacities; the candidate concerns request-row accounting, not exhaustion of all state slots. Sampled .27 Mamba usage and approximately .10–.11 full-token usage do not support a sustained capacity-pressure explanation; they do not measure every transient reservation. Retraction is not needed for the source-derived mechanism. The candidate sets `batch_is_full` before calling the waiting request's `add_one_req`, so an actual KV `NO_TOKEN` response is not required.

Prefill delayer is disabled, minimum-free-slot delay is null, prefill/decode interval is zero, dynamic chunking is disabled, `prefill_max_requests` is null, speculative decoding is absent, and priority preemption/HiSparse/HiCache/LoRA/disaggregation are not active in the resolved state. The AMD-only tile gate returns immediately on a non-HIP runtime (`schedule_policy.py:606`), so it does not explain this CUDA run. Native nonmixed chunks prioritize prefill and pause ordinary decode (`scheduler.py:3446`, `:3806`), which explains some prefill interference but does not alone explain an approximately 20-second interval of three decoders with a waiting fourth.

Overlap schedules the next batch before processing the previous result (`scheduler.py:1864`, `:1898`), so occupancy and finished-request observations may lag. However, the row reuse plus persistent full flag provides a concrete path without inventing an overlap-reserved slot or a hidden permanent three-request cap. No saved trace records all actual row owners or the full-flag writer, so alternate paths setting the same flag cannot be eliminated conclusively.

This investigation does not resolve C1 kernel efficiency or predict the performance of a corrected scheduler. Triton W8A8 versus Marlin remains a separate question.

## One future controlled test, not executed

After the baseline campaign, repeat one identical C4 cold 8192/512 wave with **diagnostic logging only** in a separate experimental run, retaining version/config/inputs. Log at the full-flag write at `scheduler.py:3656` and the early return at `:3551`: iteration and request IDs, prior/new `batch_is_full`, `running_bs`, `len(can_run_list)`, row IDs and `holds_kv` for its members, pool `size`/free rows, PP budget, beam pending rows, `chunked_req` before/after `add_chunked_req`, waiting count, and a reason code for every other full-flag write / `AddReqResult`. Also log resets on batch filtering. The decisive expected record is a reused final-chunk row with `(running_bs, can_run_count, free_rows)=(2,1,1)`, followed by full=true with running3/free1/waiting1 until a completion resets it. If that is absent, inspect the recorded alternative full-flag reason rather than asserting this candidate as confirmed. No optimization or configuration change is proposed for the current baseline.
