# Full64-layer gate/up and down fixed GPTQ

All128 matrices completed under original owner114333/start44153422; child114337/start44153444 returned0 and was reaped. Original FP8 logical weights, fixed H512/G512/staticamax/7, GPTQ FP64 damp.01/noactorder and independent WikiText4train+2validation requests. Fit and hashes freeze before each layer validation read. Offline CUDA training is not inference timing.

All128 fixed CPU-selected row code comparisons:0differences. L31 full gate/up178257920 and down89128960 codes, plus their scale words, equal prior CPU-fitted021/020 exactly. Cache packed codes+scales totals8690073600B. Model startup fit remains pending026.

Heldout linear L2 improves versus frozen minmax in64/64 gate and63/64 down matrices; down L3 worsens. Gate median .156762→.141389; down .170250→.161126. This is a sampled-row calibration diagnostic, not task accuracy or an acceptance threshold. No task-score qualification is claimed.

Compact sealed metadata/source/command/log packet: true-int4-all-layer-GPTQ-025-review.tgz,468502B,593members,SHA256371299221eaaae0cc288617c1a66c6c4a6c58fbafd3b53b010a8d9306b79ba8c. Large original activations and code/scales remain at the original remote paths identified perfile; no original weights were modified. Current queued continuation:027tail fullraw/Graph→026allFFN model→029actualcapture complete arithmetic replay→028r2untracedC16.
