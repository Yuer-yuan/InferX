| u3 client14/native13 | TTFT P95 (s) | TPOT P95 (ms) | Output TPS | Result |
|---|---:|---:|---:|---|
| C14-native13-080 | 10.237349 | 40.371241 | 308.277670 | COMPLETE_JOINT_FAIL |
| C14-native13-bounded100 | 9.893643 | 39.929242 | 306.038803 | COMPLETE_JOINT_PASS |
| C14-native13-bounded100-r2 | 9.821661 | 39.141798 | 309.951742 | COMPLETE_JOINT_PASS |

All three original128+8+1 runs retained; zero errors and matching original payload salt. P95 and TPS independently checked from profiles/actual phase times. Both100 repetitions pass, but first TPOT margin is only0.070758ms. This normal rebuild has no established GPU cubin equivalence to080; task accuracy for100 remains unmeasured. See ../summary-comparison.json for exact identities, phase spans, linearP95 brackets and process terminal evidence.
