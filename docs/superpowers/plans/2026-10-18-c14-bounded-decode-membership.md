# C14 bounded decode membership

The current goal is C13/C14 joint TTFT<=10s and TPOT<=40ms using complete original128 requests, while retaining passed configurations. Uniform completion margin3 failed; do not combine it with this candidate. Parent is frozen080 ordinaryK3, sourcea6acc1d2, with all original caches/operators and public budgets unchanged.

Measured component098 establishes same-membership eager/deferred correctness and a cost tradeoff: full14/subset9/full14 takes156.733ms versus162.728ms for three full rounds, but has different progress. Selected-nine progress rises6.17%; omitted-five progress falls33.57%; total cost per committed token worsens11.19%. This is service redistribution, not an overall acceleration. It warrants at most one bounded urgency-policy trial, not blanket batch reduction.

Candidate algorithm (default OFF; one fixed policy, no grid):

1. Keep the original full-membership PF/decode decision and all hard PF boundaries. Apply membership selection only when the original loop actually chooses to run decode.
2. Only consider current full-ready B10..14, covering K3 targetT40..56 to selectedT36 within the same qualified outputS4 domain. Other sizes preserve full membership. Require an observed full-B cost bin, finite positive costs/yields and valid request completion views.
3. Compute each request's original whole-completion laxity with40ms/token and the existing max(batch-yield rounds, own-yield rounds). No predicted future request identities, test-size exceptions, or hardcoded generated-token counts.
4. Select the nine least-lax requests only if the minimum laxity is negative and every omitted request has at least three current full-round costs of positive laxity. Otherwise keep the full set. Tie-break deterministically by lane and preserve original lane order in the resulting membership.
5. After a subset, force the next two actual decode rounds to use all ready requests. PF work does not consume this count. This bounds skipped service and matches the qualified full/subset/full cycle structure. No cancellation, completion, sampling, or output-delivery behavior changes.
6. Cost observations and committed_decode_rounds must use actual selected membership. Do not count omitted rows as serviced. PF policy continues to use all ready requests and the original full-B cost.

Root's initial CPU analysis of the current095 trace, on observed full-decode states only, finds187 eligible states with this policy. Current trace percentile engine99/126 are selected11/30 times and omitted0 times. Formal094 tail session019, engine19 in095, is selected41 times and omitted2. These are not a simulated schedule, additive savings, or predicted P95; changed arrivals and membership can still worsen performance.

Implementation split: u3 worker prepares a pure CPU selector/header and registered tests from frozen080; u2 owns host integration, normal build and necessary state-boundary numeric qualification; u1 independently reviews selector and PF-interleave/cancel/lifecycle boundaries, then runs the original128 C14 on the same card as formal094 after Root relays the frozen binary and verifies dependencies. Root freezes all identities and checks evidence. No edits to original frozen source/binaries. No GPU-object swapping.

Required checks: OFF exact behavior; insufficient observations; B bounds; all-safe/no-urgent and unsafe-to-omit cases; stable ties and row order; exactly9 unique live rows; two-full cadence including an intervening PF; cancellation/membership changes; nonfinite inputs; actual statistics count only served rows. Inspect whether PF between subset and next full introduces a state path not exercised by098; if so add a focused same-membership native eager/deferred gate before HTTP.

Run only after qualified build/CPU/native boundaries: original080 settings plus one enabled selector flag, K3/native14/KV121856/PF2048/traceOFF, exact original128+8+1 payload including salt. Compare against formal080-K3 C14. Any failure stops this fixed-policy trial; no promotion or lower-tier borrowing. PASS requires same-binary C13 then C12/C8/C4 regressions, and original100 quality last. Preserve failures and old working artifacts.

Parallel configuration experiment: on idle u3, frozen080 ordinaryK3 may serve client14 with native13 and KV121856. This keeps all original client requests/arrival behavior and charges admission waiting to TTFT. It must be labeled client14/native13, not conflated with native14 comparisons. Official local runner measures client concurrency; prior native=client was the controlled default, not a requirement to alter the client load. Verify effective47-layer output cache and all478 serving files against the original full512-file catalog before launch. No native-cap grid.
