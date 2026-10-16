# QoServe policy extraction evidence

This is source/CPU evidence, not a model performance result. The author is
`microsoft/sarathi-serve`, branch `niyama_asplos2026`, commit
`72339db5b120614033b317880ad3a05a5e4c9415`. The parent source capsule contains the
MIT license and all fourteen pinned files; the separate timing file is bound
by its actual bytes in `manifest.json`.

`author-policy.py` is reproducible with `inferx.qoserve_policy.load_author_policy`.
It contains the original predictor formula, chunk search, sequence ordering,
cumulative TBT deadline, tier budget, one-time requeue and throughput EMA.
The manifest records hashes of original source fragments and generated methods.
The two capacity reads are the explicit native boundary; no old cache-manager
API is simulated. The later native dispatcher is separate compatibility code.

These saved coefficients are the author's reference A100 tables. Calling
`require_deployment_coefficients()` on this class raises an error. Supplying
hardware tables permits that presence check, but profile identity, independent
holdout coverage, completed real-model qualification and efficacy remain later
gates. No QoServe GPU campaign has run.

Fifteen local policy tests exercise literal numerical expectations and source
integrity. The first run failed because the loader was absent, then all fifteen
passed after extraction. An initial generic-suite attempt accidentally used
uv's default Python3.10, producing43 environment failures (including missing
`asyncio.timeout`/`os.pidfd_open`); the corrected run explicitly uses system
Python3.12. A subsequent uv overlay run had one existing repository-verifier
failure: its sandbox allowed the overlay's site-packages but pytest's actual
modules were in the cached environment. Running that environment's Python
directly resolves the path mismatch; the failing verifier test then passes.
No restriction was weakened. Complete logs retain these attempts.
