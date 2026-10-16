# QoServe collector and fitter verification

This directory records implementation gates, not SOTA efficacy measurements.
The real-model profiling campaign uses immutable source v5 on the HDD of
vlm-server; v6 additionally contains the separately tested fitter and native
solo-budget collector.

- `native-profile-v1-red.log`: four missing-profiler errors, old native tests pass.
- `native-profile-v2.log` and `native-profile-v2-geometry-diagnosis.json`: the CPU
  fixture had cache block16 versus logical/hash/checkpoint256. Actual scheduled
  work was112+1, correctly recorded by the collector.
- `native-profile-v3.log`: corrected fixture,18 actual native tests pass.
- `native-profile-v4.log`:19 actual native tests pass, including EngineArgs and
  current EngineCoreRequest ABI/salts. These CPU tests ran before GPU timing.
- `fit-v3-red.log`: seven scientific contracts actually run and fail for missing
  fit_profile. Earlier dependency-missing skipped attempts are not RED evidence.
- `fit-v5-green-attempt.log`: six pass, one incorrect synthetic rank expectation;
  the lower fixture also has D=(total-128)/128. Independent centered matrix ranks
  are3/4, matching the unchanged original OLS.
- `fit-v6-green.log`: seven scientific contracts pass with the five pinned
  scientific dependencies. `scientific-all-qoserve-suite.log`:48 pass,19 native
  tests skipped here and run separately as recorded above.
- `collector-budgets-final-suite.log`:363 generic tests pass,40 skip (14EB native,
  19QoServe native,7scientific). Scientific tests did actually run separately.
- Other RED logs cover fixed protocol/salt isolation, exact restoration identity
  and argv, common independent-budget quantiles/tags, and client-to-engine joins.

Full-file local scientific-wheel downloads repeatedly restarted. Their retained
logs are interrupted attempts, not successful installs. Ranged downloads then
verified the official PyPI wheel hashes; the final private local environment
matches the already validated private HDD fitter environment. The native
vLLM/Torch environment was not changed. No heavy remote CPU work was run during
the GPU profiling campaign.

`source-task3-v{4,5,6}.tar.gz` preserve source bytes. SHA256.json inventories the
artifacts. The independent native model-budget campaign and formal QoServe
comparison have not been run by this checkpoint.
