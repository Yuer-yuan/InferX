# 上下文与客户端接入：执行决定及最终审查

任务以用户授权的原工作区、原研究分支继续。直接使用入口见 [使用说明](../agent-quickstart.md)。下列记录保留实验决策、代价、测试和审查处置。

## 延后的小项

当前运行服务的 runtime.json 和 agent-server.log 是本次部署记录。现有启动器在以后重启时不自动更新这两处；前台输出在终端，未显式设置 INFERX_REQUEST_LOG 时请求日志采用 tuning/ninfer-requests.jsonl。重启后的进程记录仍需手工核对和更新。未在本轮增加服务生命周期管理。

## 执行账本

```text
# SDD ledger — plan: docs/superpowers/plans/2026-10-14-context-agents.md
Start: 7a967f8; existing research/ninfer-mtp-tuning branch and user-selected work directory.
Ruling: Proceed with authorized context expansion and three local clients without repeated approval — user explicitly chose this work and repeatedly asked to continue; developer authorization overrides additional skill gates — cost if wrong: reversible HDD client artifacts and owned GPU1 service, with64K rollback.
Ruling: Keep inline execution/current branch — preserved session preference; no push/merge — cost if wrong: later integration remains manual.
Ruling: Choose128KINT8 then256Krk4v4-e8, MTP7,C1 — reuse supported engine paths and trade KV precision for context only when validated — cost if wrong: failed candidates or task-quality degradation remain measured, rollback64K.
Ruling: Use new Pi namespace1.2.0/Node22.23.3 and pinned OpenCode1.19.0; reuseClaude2.1.92 — upstream renamed and existing Node18 is too old — cost if wrong: isolated version rollback, no global changes.
Pre-flight: Task1 runtime manifest -> Task2 client budget/local routing: declarations must use selected measured context.
Pre-flight: Task2 wrappers -> Task3 runner: common executable interface, fresh task copies and serial use preserve C1 interpretation.
Builder check: fresh context_agents_design_check returned nothing important left.
Task1 RED: missing inferx.context_agents module, pytest collection failure observed; implemented identity/budget/retrieval helpers, targeted3passed.
Task1: generated lookup prompts122879/253914tokens usingGPU0 read-only tokenizer, reserved8192outputtokens. Selected early/middle/late pseudorandom secret lookups.
Ruling: overlap independent Task2 package download with Task1 GPU validation — isolated HDD installation does not consume inference GPU or change context — cost if wrong: noisy host load recorded, not a paired engine performance claim.
Task1: verified priorPID2675225 owner/exe/start ticks, usedpidfd SIGTERM to stop it, archived64K rollback. 128KINT8probe launched with1sGPU telemetry.
Task1:128KINT8 passed122879-token early/middle/late lookup, observedminfree1443MiB/maxused22665MiB;256Krk4v4-e8 running. Both prompts reserve8192tokens. 128Kheldout3x complete.
Task2 preparation: Node22.23.3 checksum verified, OpenCode1.19.0/Pi1.2.0 installedHDD. --ignore-scripts caused OpenCode binary missing (actualversioncommandRED); inspected official postinstall source (localbinarycopy, TMPDIRHDD), ran onlyitsbinarysetup; versionsGREEN. GlobalNode18/Claude2.1.92 unchanged.
Task2 configRED: missing agent_configs import observed; local-only budgets/alias tests implemented,2passed. Provisional128K config parses inOpenCodeandPi; will regenerateafterselection.
Task1:256Krk4v4-e8 passed253914-token early/middle/late lookup; minfree1307MiB,maxused22801MiB. All12 shortcode outputs128K/256K pass same sandboxchecks. Choose256Kexpandedagentserver;128KINT8 and64K rollback remain.
Ruling: prepare common agent config/process/verification helpers while GPU probes run — independent CPU work reduces idle time and keeps interfaces in place — cost if wrong: final whole-branch review checks combined changes; formal tasks still record their own deliverables.
Task3 helpersRED: missing inferx.agent_trials module observed; bounded owned-group timeout/non-success classifier GREEN. Added real pristine-test tampering regression: wrong production source still fails despite agent-side rewritten tests, targeted3passed.
Task2: native Claude2.1.92 binary confirms auto-compact/context/output/config-dir variable names; configurations stay outside user-global profiles. No adapter or account login yet.
Task1 suiteRED: existing EpisodeTests.test_waiting_deadline_cleans_permit_for_later_arrival raced equal deadlines; waiting admitted~38ms thencancelled~40ms, not a leakedpermit. Added deliberate first-generator cancellationcleanup delay toholdpermitpastbothdeadlines and widenedlaterarrival/makespanmargin; assertions preserved, full83testsGREEN.
Task2 initialversionchecks created3emptyOpenCode defaultdirectories and4bytesPidefaultstate outsideHDD. Birthtimes/owners/contentsizes prove task-created, movedonlythosenewpaths toHDD initial-client-state. Installer now sourcesagent-env beforeversionchecks; all futurestateHDD.
Task1 warm256K PID2726457 verifiedowner/exe/startticks/cwd; /v1/models reports262144. No coreengine changes. Fullsuite83passed,12codeoutputs pass.
Task 1: complete (commits 9bf538f..2da8fb1, tests: ssh vlm-server 'source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh; cd "$INFERX_ROOT/code-tuning"; "$INFERX_ROOT/repo-venv/bin/python" -m pytest -q' → 83 passed in 5.48s)
Task2: all3nativeclients completed realread/edit/bash testsmoke; pristine smoke tests verifiedagain, logs/clientversions archived. No protocoladapter needed.
Task2 CLI RED: Claude variadic --tools consumedprompt; added -- delimiter, same smoke GREEN. AllClaude requestsnativeqwen38-27b/max_tokens8192. SDK cloud-cost field is an estimate, no cloud endpoint used.
Ruling: Claude2.1.92 reports unknownmodel200K/32K metadata — use conservative200Keffectivewindow (backend256K), actualrequests capped8192 and localaliasesfixed — cost if wrong: verylongClaudetraces need additional metadata/compaction validation; do notclaimfull256Kclientwindow.
Task3 stdin RED: child inherited controllerstdin and consumedshellcommands; realpiped sentinel test failed; stdin=DEVNULL fix,5helpertestsGREEN.
Task3 parser RED: missing parse_agent_events; implemented native completedtool/terminal/error parsing withIDdedup; targeted5testsGREEN. Pendingtoolcalls cannotproveexecution.
Task 2: complete (commits 2da8fb1..dd48a49, tests: ssh vlm-server 'source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh; cd "$INFERX_ROOT/code-tuning"; "$INFERX_ROOT/repo-venv/bin/python" -m pytest -q' → 85 passed in 5.63s)
Task3: nine serial repository trials launched, frozenfirstdevtask perfirst3repos (604/json_repair,1155/xmltodict,33/PTable), privatecopies,ablationmustfailpristinetests beforeagent. Nativecompletedrequests/token/cache counts archived; controllers haveDEVNULLstdin; verifiers usepristinetests/sandbox, onlytargetsourcechanges count.
Usersteering: askedaboutkvcachedmanualKVmanagement. Read primaryovg-project/kvcached and actual NInferarena/pagepool; currentNInfer haspagedquantizedlogicalreusebutLinuxphysicalpoolpreallocated (~4.84GiB persistentstate for256K). kvcached now supportsQwen3.8hybrid invLLM/SGLang, noNInferadapter listed. Answered incommentary; do notreplaceengineorimplementKVhook without evidence/explicitnewscope. Add allocation/cache observations to report while preservingactiveagenttask.
Task3 initialunrestrictedattempt aborted: OpenCode ignored offline prompt, downloaded upstreamwheel into/tmp/jr9. Verifiedcontroller/ownedchild identities, stoppedonlyownedgroup/controller, archivedallinitiallogs/copies outsideofficialtrialglob; movedtask-created/tmpdownloadHDD afterbirthtimecheck. Noresultcounted.
Ruling: enforce task-only inherited Landlock filesystem rules and TCP-connectports18081/18082 — actual policy violation showspromptalone insufficient; preservewriteaccessonlyprivateworkspace/HDDstate/cache/tmp, preventreadingcanonicaldataset — cost if wrong: clientcompatibilityrestrictionmaycauseinfrastructurefailures, whichareseparatelyreported. This is a non-adversarialresearchboundary, nothost/IP-specific networkisolation.
Task3 restrictionRED: missing agent_restrictions; realchild test proves workspacewrite/APIportallowed, outsidewriteandotherTCPportdenied, GREEN. All3restrictednativeclients repeatread/edit/testsmoke passed. Official9trialgrid restarted withrestrictionenabled and unchangedtasks.
Task3 progress: official run session43930/remotePython scripts.run_agent_trials active. /604 OpenCode timeout600.08s/read3/bash310/no_source_change/pristinefailed, native313completedrequests,maxprompt92301,cachefraction.997. Pi/604timeout600.03s/read5/bash336; Claude/604no_source_change35.75s. OpenCode/xmltodict1155passed47.19s/read9/glob1/bash4/edit4 withoriginal60tests. Pi/xml1155running. RemainingClaude/xmlandall3/PTable33. Keep600s limit and unchangedtasks; no generation-guided selection or hidden failed-trial removal.
Task3 integrity: inspectedactuallastBashcommand/result, codeexec succeeds andreturnswrongJSON; repeateddiagnosticloop isnot attributed toprotocolescaping. AllAPIrequests local, idleGPU service remainshealthy. Full86testsGREEN afterrestrictions. Additionalinitial-agent/tmp/jr*downloads tracedfromrawcommands andbirthtime/UIDverified, movedHDDarchive (notformaltrialresults).

Usersteering: defer KV management; provide immediately usable OpenCode/Pi/Claude Code configurations, locations and rationale. No new KV experiments.
Task3: all9 official trials finished before stop check; no benchmark process remains, backend idle. OpenCode2/3,Claude2/3,Pi0/3 (three600s timeouts). Allfailures retained; native request/cache observations do not explain task-quality causally. Rawmanifest35files,3x3grid audited.
Ruling: prioritize concrete direct-use handoff and remove KV research from report — latest user explicitly deferred KV management — cost if wrong: KV mechanism work remains unimplemented and can be resumed separately.
Task3: quickstart/configpaths/rationale committed with fullresultreport; 256Kservicehealthy, GPU0 preserved, CLI versions1.19.0/1.2.0/2.1.92, full86tests passed5.76s andbashsyntaxpassed.
Task 3: complete (commits dd48a49..cf5d2e4, tests: ssh vlm-server 'source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh; cd "$INFERX_ROOT/code-tuning"; "$INFERX_ROOT/repo-venv/bin/python" -m pytest -q' → 86 passed in 5.72s)

Final review: fresh gpt-6-astra context_agents_final_review, read-only range7a967f8..cf5d2e4. NoCritical. Important: exporterglob includeditsowncompactoutput onsecondrun. Rootcause reproducedwith2realCLI regressiontests RED (results doubled; duplicate rawidentity accepted).
Final: fixed repeatedexport/duplicateidentities — test_repeated_export_preserves_nine_trials_and_aggregates and test_export_rejects_duplicate_raw_client_task_identity RED→GREEN; full88/88passed5.82s. RealHDDexporttwice givesidentical3artifacthashes and9results. NoadditionalGPUexperiments.
Final: minor (deferred): start_agent_server.sh doesnotrefresh current-session runtime.json/logpaths onrestart; current running deployment matches advertisedpaths, later foregroundrestart usesunderlyinglauncher defaults and requiresmanualrecordupdate.
Final: Ruling: reviewer setaside newKV/enginechanges — userexplicitlydeferred them and nativeclientAPIalreadyworks, so keepexistingengine — cost if wrong: no newKV mechanism delivered.
Final: Ruling: reviewer setaside adversarialescaperesistance — retain documentednon-adversarial benchmarkboundary, interactiveclientsuseordinarypermissions — cost if wrong: boundaryisnotanadversarialsecurityguarantee.
Final: Ruling: reviewer couldnotinspectremoteprocesses/binaries/rawlogs — executorverifiedownedPID/startticks/exe, actualCLIversions, bothhealthports, confighashmatches and35rawhashes; accepttheseobservedchecks — cost if wrong: evidenceispoint-in-timeandmustberecheckedafterrestart.
Finish: keep research/ninfer-mtp-tuning branch/workspace andliveGPU1service aspreviouslyauthorized; no push/merge. Archiveledgerbeforecleaningonlythisplanscratch.

```

## 最终审查结论

独立 gpt-6-astra 审查未发现 Critical；发现 Important 一项（导出器重复读取派生结果），已用两个真实 CLI 回归修复，完整88项测试通过。原始九次试验不需重跑，实际结果连续导出两次完全一致。Minor 一项（重启后日志及运行信息不自动刷新）延后，见上面的说明。

审查未判断的范围及执行者裁定均记录在账本 Final: Ruling 行：新 KV 机制暂缓；实验边界保持非对抗性；远程现状由执行者实测核对，证据仅代表观测时状态。
