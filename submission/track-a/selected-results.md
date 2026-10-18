# 当前100版本：性能结果与运行配置

冻结100版本已通过C4、C8、C12、C13以及两轮C14。各轮均为原128条正式请求、8条预热及1条探针，冷输入8192 / 输出512 token，128成功、零错误。SLO要求TTFT P95≤10秒且TPOT P95≤40毫秒；排队时间包含在TTFT中。

| 节点 | 客户端/执行槽 | 轮次 | KV容量 | TTFT P95 (s) | TPOT P95 (ms) | 输出token/s | SLO |
|---|---:|---:|---:|---:|---:|---:|---|
| ucloud3 | 4/4 | 1 | 65536 | 2.269117 | 22.891142 | 205.8182 | PASS |
| ucloud3 | 8/8 | 1 | 69632 | 5.921628 | 29.532080 | 274.2038 | PASS |
| ucloud1 | 12/12 | 1 | 104448 | 9.593671 | 36.234647 | 299.7572 | PASS |
| ucloud1 | 13/13 | 1 | 113152 | 9.706068 | 38.181167 | 302.9681 | PASS |
| ucloud3 | 14/13 | 1 | 121856 | 9.893643 | 39.929242 | 306.0388 | PASS |
| ucloud3 | 14/13 | 2 | 121856 | 9.821661 | 39.141798 | 309.9517 | PASS |

C14以14个客户端请求并发测量，引擎最多同时执行13个；两轮原始结果均保留，首轮TPOT余量仅0.070758毫秒。单轮或两轮通过不等于长期稳定性保证。C4/C8/C14与C12/C13来自不同节点，不能把整条曲线解读为只改变并发的同机受控实验。

同一程序C14/native14为9.724916秒 /41.022342毫秒，联合FAIL，见[原始汇总](../../results/three-node-optimization/ucloud1/bounded-decode-C14-096/analysis-C14/summary.json)。C15/native13为9.960209秒 /41.881277毫秒，联合FAIL；C16/native13完整测得TTFT13.275426秒、TPOT42.327509毫秒、310.6298token/s，联合FAIL，见[原始汇总](../../results/three-node-optimization/ucloud3/c12-c16-latency-20261018/C16-native13-bounded100/summary.json)。C16/native16为9.967466秒 /51.220646毫秒，TTFT改善但TPOT恶化，仍联合FAIL，见[同卡容量对照](../../results/three-node-optimization/ucloud3/c12-c16-latency-20261018/C16-native-capacity-comparison.json)。C4第一次启动因KV34816低于context65536在任何请求之前被CLI拒绝；表中是合法KV65536的完整测量，原失败启动单独保留。

同卡C12相对重新测量的9270，TTFT从9.600909降至9.593671秒、TPOT从38.749242降至36.234647毫秒；C13的TTFT约增加21毫秒，两版均满足SLO，不能声称所有指标逐项无退化。100是正常新构建，尚未建立其GPU cubin与旧080完全相同；不把组合效果归为纯host策略收益。

## 精度

同一100二进制已完成原IFBench前100题评测105：native/client13、KV532480、原111条约束、默认thinking与模型采样、自然EOS及max32768。严格题目评分 **85/100**，宽松 **88/100**；严格指令评分 **96/111**，宽松 **99/111**。100次请求全部成功且自然停止，零截断；**1条最终答案为空，保留并计入评分**。原583个文件逐项SHA核验，并从原HTTP答案独立重跑原verifier，100题结果与汇总完全一致，见[105完整报告](../../results/three-node-optimization/ucloud2/frozen100-flag1-C13-quality-preparation-105/REPORT.md)。

题目strict/loose的95% bootstrap区间分别为[78%,92%]、[81%,94%]。旧6632版本同100题为84%/87%，这两次观测没有显示分数下降，但未建立同配置的高精度对照或稳定性保证；100的质量不能用于正在开发的W2组合版。

## 复现

从[100运行入口](../../demo/selected100/README.md)部署匹配模型、缓存和动态依赖。该目录自带冻结原runner、原137条数据及两份显式资产配方；服务与客户端必须选择同一配方。所有运行目录独立创建，脚本不会覆盖历史结果。

- binary SHA256：`e4ed1bf4baebf7da3a81e4aed01307c7e3e7ae97faef7672c507ed21bdaeb38c`。
- source SHA256：`c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`。
- 普通MTP K3、PF2048、Graph开启、bounded membership开启；19个NINFER变量固定于recipe。
- 性能KV为`max(65536,8704*client_C)`；C14明确为native13/KV121856。
- [全部原始来源与精确数值](data/selected-campaign.json)、[六点独立核验记录](../records/selected-campaign/README.md)。

入口在当前路径通过13项CPU测试；已有24次搬迁dry-run。它们证明脚本契约与路径可移动性，不等同于在新硬件完成GPU复测。大型模型/cache内容SHA继承冻结清单，启动时重新核验binary SHA及资产大小/集合；同尺寸资产损坏不能由轻量检查发现。
