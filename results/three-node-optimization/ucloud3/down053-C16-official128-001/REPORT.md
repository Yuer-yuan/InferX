# dc99 C16：完整128，仍未通过

| 同机C16 | TTFT P95 | TPOT P95 | 吞吐 |
|---|---:|---:|---:|
| 9270原版 | 9.934084 s | 51.702542 ms | 307.147754 tok/s |
| dc99 + down053 | 9.907686 s | 54.112965 ms | 311.478293 tok/s |

本轮128测量+8预热+1probe全部完成，0错误、137cold8192/cache0/full_reset。136条512输出，原probe1输出；startup全保留。固定K3/native16/KV139264/PF2048/450W/seed20261016/traceOFF。binary dc99ec83…694424，source f592cb8d…af6198；实际55依赖全SHA通过。

实际137payload除cache_salt完全相同，warm8/measured128分别同cohort，SHA `62f94272362623a9d776cf99711ab67411957a1d7799b7987583b2570df4b021`。actualenv仅新增NINFER_DOWN_H512_SINGLEWARP=1；其余量化/调度/权重不变。所有P95本地逐行独立复算吻合；吞吐保留原phase值，并按phase时间精度核对。

吞吐变化+1.409920%，TPOT P95变化+4.662098%；C16仍FAIL，不晋级。它是一次同机顺序对照，不是稳定收益证明。旧9270 C12的本轮PASS保持原结论，未测dc99 C12。

全部4个owner/driver/telemetry/service actualmissing且GPU空；实际终态、原client/inputs/native/1HzGPU/配置/qual/失败信息在delivery/raw，53文件逐SHA验证。raw包8,750,715B，SHA cb8231b25ae001e0a22d7fc5ecb884579bb3853087e01c1a2fbcbfa33b685437。summary.json含精确指标、个体违规和P95贡献请求；measured-request-index.jsonl保留128行。没有追加profile、seed、C12或精度任务。
