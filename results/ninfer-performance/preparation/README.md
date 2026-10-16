# 部署与接口核验证据

工具和配置按功能命名；子目录 `ucloud1/2` 仅表示这轮证据来自哪台机器。机器原始盘点位于相邻 `inventory/`。实际评测只在第一台运行，第二台用于后续并行验证，不重复产生比赛成绩。

两台使用相同固定 NInfer v2 权重与原生构建，SHA256 分别为：

- 权重：`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`
- 构建：`b604d0507e753de76e6b3b288011434b2bb07f9038738f8c73768ddac8dfba86`
- completions 补丁：`d5534f0189f1e05c91dc32d5ca6d19541d52ca94f7f95209eb009193b2fe87d1`

第一台从 hf-mirror 下载并完整校验权重、编译原生 CUDA 服务；第二台接收公开权重和同一构建，重新校验 SHA256。两机 GPU 型号、显存和驱动相同，CPU 型号不同，后续改动须相对各自同机基线比较。

第一台真实探针通过模型 tokenizer 的8192输入/512输出、raw string与token-ID、流式/nonstream一致、ignore_eos、准确usage、零缓存复用及非法请求拒绝。第二台只做32/16短探针、chat、add工具调用和结果续接，身份核验后重启，再通过8/8短探针。两台原版 judge依赖检查都ready；不据此宣称正式性能或IFBench通过。

`config/` 保存模型验证记录、客户端pip版本和带校验和的Conda explicit lock。后者通过`micromamba env export --explicit`生成，第二台完成lock解析dry-run，尚未据此宣称全新环境重建完成。`qualification/`及`restart/`保存实际响应、服务日志和启动argv。NInfer资源提取在第二台重新核对全部六个嵌入资源SHA，清单保存在该节点config中。

这些服务PID/时间为**当时的快照**，不应拿它们直接控制当前服务。当前后台句柄位于节点`$INFERX_ROOT/config/active-service.json`；控制前仍须核对身份。实验端口切换遇到的TCP TIME_WAIT在`restart-port-state.json`中保留：state06、无监听者、bind errno98。实验脚本已为本轮已回收服务增加有界等待，并继续立即拒绝外来监听者。

原始日志以无损gzip保存，`compressed-log-manifest.json`记录压缩前后SHA256，未清理日志空白或修改内容。依赖下载、外部源可用性和overlay持久性不构成推理性能结论。部署与运行步骤见`submission/REPRODUCE.md`。
