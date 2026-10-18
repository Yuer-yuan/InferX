# 9270：C16 未通过，C12 本轮通过

| offered/native | TTFT P95 | TPOT P95 | 输出吞吐 | 固定 SLO |
|---|---:|---:|---:|---|
| 16/16 | 9.934084 s | 51.702542 ms | 307.147754 tok/s | FAIL |
| 12/12 | 9.499646 s | 37.340162 ms | 284.746019 tok/s | PASS |

每轮均为原 runner d80、128 测量 + 8 预热 + 1 原始 probe，零错误。137 条 native 全为冷8192/cache0/full_reset；136条输出512，probe输出1。所有 startup 保留。固定9270/source91621、K3、chunk2048、450W、seed20261016、traceOFF。C16 KV139264；C12 KV104448。

本地逐行重算两项P95与原parser吻合。吞吐保留原profiling指标，并用65536token与真实phase起止时间核对；导出时间仅微秒精度，不能误用另一个benchmark_duration跨度。首次CPU错误公式断言失败另存，没有改原统计或门限。

137份实际输入payload除cache_salt完全相同；8预热及128测量cohort分别相同。canonical payload SHA：`62f94272362623a9d776cf99711ab67411957a1d7799b7987583b2570df4b021`。这是并发及KV资源设置对照，不是单算子因果A/B。

C16个体TTFT违规6条、TPOT32条；C12分别2条（TTFT arrival10/12，TPOT19/106）。P95贡献请求和插值权重见independent-local-raw-audit.json，全部256条测量见measured-request-index.jsonl。C12只称本轮通过，不称独立seed稳定。

C12首次port18486 preflight报EADDRINUSE，无service、无请求，raw/C12-port-failure完整保留。recovery仅改port18487与独立输出/receipt路径，未重跑C16。原外层job失败状态未覆写，C16已完成测量单独保留。

全部10个owner/driver/telemetry/service实际missing，子进程wait/reap及GPU空闲见raw/actual-terminal.json。完整client/input/native/1HzGPU/config/身份/失败证据114文件逐SHA验证，raw包21,158,205B，SHA837fb0d63638903f5e105c8503359bb8eda6d498df0f5e707eee6377ca0ca4e2。

先前原100质量按用户要求取消，5完整响应/7attempt保留，不是100成绩。旧protectedT4 A16逐位失败亦保留，新selected47路径模型实际门单列。无新质量资格、GPU关键路径、occupancy或跨seed稳定声明。本轮完成，无追加旧binary任务。
