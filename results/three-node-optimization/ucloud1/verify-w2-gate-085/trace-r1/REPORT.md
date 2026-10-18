# 085 单次调度诊断

同binary14d32144/sourceb2343/cache63a48f、K5/native16、原128+8；唯一模型环境差异NINFER_SCHED_TRACE，实际输入连salt完全相同。137native冷请求/0error；trace138请求多出的内部startup为53PF/4output，另probe8192/1，均保留。所有owner/child已退出。

51原始文件SHA通过。Linux单调/epoch起止锚点最窄140/130ns（双精度换算限制另存原整数），128/128请求按绝对first/end既定5ms阈值唯一关联，非header ID join。550PF+1275decode host units没有重叠；按完整128先选P95，未对子集重算。

诊断TPOT P95 49.39804ms = 其他请求PF31.93407 + 包含该请求decode17.45229 + 不含该请求decode0 + control/未记录等待0.01168。这里是executor host wall跨度，不是GPU独占。官方latency metric与metadata end微小差异单独存client_metric_bookkeeping_correction_ms并归残差，不延长任何unit或放宽join。原未校准CPU分析保留r1-unadjusted，未改raw。

诊断TTFT P95 10.00953s = ownPF0.90860 + 其他PF9.07545 + decode0.02365 + residual0.00183。

测量生命周期窗1396次选择中conflict257（18.41%）；所有选择PF的508次中50.59%为conflict。282次队列latest_start已为负：261次多job、21次仅head。该字段未直接记录limiting job索引；多job负值不等于队尾必为原因，Root独立重建队列/成本。

原traceOFF尾session52本次仍50.22959ms，PF17.2775s、参与decode8.3838s、其余6ms；session108本次38.16626ms，PF14.1095s、参与decode5.3894s，说明同输入仍有闭环时间轨迹变化。正式085原128FAIL不被诊断替代。

结论：此轮尾部remainder主要是其他请求PF插入，不是非参与decode或主机空白。调度冲突值得针对性修复，但不凭诊断承诺SLO收益，不接参数网格。
