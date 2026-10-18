"""Portable CPU contracts; exec, HTTP and evaluation execution are forbidden."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import candidate_common as common
import serve_candidate
import perf_candidate
import accuracy_candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"relocated root with spaces"
        self.root.mkdir()
        self.path = PACKAGE/"candidate_recipes/100-k3-u3-47.json"
        self.recipe, self.sha = common.load_recipe(self.path)
        for target in ("os.execvpe", "urllib.request.urlopen", "subprocess.run", "subprocess.Popen"):
            patch = mock.patch(target, side_effect=AssertionError("Runtime work forbidden in CPU test"))
            patch.start()
            self.addCleanup(patch.stop)

    def capture(self, function, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(function(argv), 0)
        return json.loads(stream.getvalue())

    def test_all_profiles_and_explicit_inventory_selection(self):
        for inventory in ("u3-47", "full64"):
            path = PACKAGE/f"candidate_recipes/100-k3-{inventory}.json"
            recipe, sha = common.load_recipe(path)
            for c, native, kv in ((4,4,65536),(8,8,69632),(12,12,104448),(13,13,113152),(14,13,121856)):
                with self.subTest(inventory=inventory, c=c):
                    plan = self.capture(serve_candidate.main, ["--recipe",str(path),"--profile",f"perf-c{c}",
                        "--root",str(self.root),"--log-dir",str(self.root/"service"),"--dry-run"])
                    self.assertEqual(plan["profile"],dict(name=f"perf-c{c}",mode="perf",client_concurrency=c,
                                                       native_concurrency=native,kv_capacity=kv))
                    argv = plan["argv"]
                    self.assertEqual(argv[argv.index("--max-concurrency")+1],str(native))
                    self.assertEqual(argv[argv.index("--kv-capacity")+1],str(kv))
                    self.assertEqual(argv[argv.index("--draft-tokens")+1],"3")
                    self.assertEqual(plan["asset_contract"]["file"],f"assets-{inventory}.json")
                    self.assertEqual(plan["paths"]["binary"],str(self.root/"artifacts/100-k3/ninfer-serve"))
                    self.assertEqual(plan["recipe_sha256"],sha)
                    self.assertEqual(plan["selection_status"],"selected_measured_C4_C8_C12_C13_C14")
            self.assertFalse((self.root/"service").exists())

    def test_environment_matches_actual_c14_and_removes_pollution(self):
        actual = common.read_json(PACKAGE/"evidence/C14-original-environment.json")
        plan = common.render_service(self.recipe,self.sha,"perf-c14",root=self.root,log_dir=self.root/"s")
        env = {k:v for k,v in plan["environment"].items() if k.startswith("NINFER_")}
        for key in ("NINFER_S4_GPTQ_CACHE","NINFER_INPUT_S4_CACHE","NINFER_OUTPUT_S4_CACHE"):
            actual[key] = env[key]
        self.assertEqual(env,actual)
        self.assertEqual(len(env),19)
        self.assertEqual(common.clean_environment({"PATH":"p","CUDA_VISIBLE_DEVICES":"0",
            "NINFER_FAKE":"1","LD_PRELOAD":"bad.so","CUDA_LAUNCH_BLOCKING":"1","OPENAI_API_KEY":"secret"},env),
            dict(env,PATH="p",CUDA_VISIBLE_DEVICES="0"))

    def test_capacity_below_context_fails_before_launch(self):
        invalid=copy.deepcopy(self.recipe)
        invalid["profiles"]["perf-c4"]["kv_capacity"]=34816
        with self.assertRaisesRegex(ValueError,"KV capacity must be >= max-context"):
            common.render_service(invalid,self.sha,"perf-c4",root=self.root,log_dir=self.root/"service")
        self.assertFalse((self.root/"service").exists())

    def test_unsupported_tiers_and_tampered_recipe_rejected(self):
        for name in ("perf-c15","perf-c16","perf-c32","quality512"):
            with self.assertRaisesRegex(ValueError,"Profile must"):
                common.profile(self.recipe,name)
        tampered = copy.deepcopy(self.recipe)
        tampered["profiles"]["perf-c14"]["native_concurrency"] = 14
        path = self.root/"recipe.json"
        path.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError,"frozen catalog"):
            common.load_recipe(path)

    def test_performance_plans_keep_client14_and_frozen_payload(self):
        for c in common.TIERS:
            plan = self.capture(perf_candidate.main,["--recipe",str(self.path),"--root",str(self.root),
                "--concurrency",str(c),"--output-dir",str(self.root/"perf"),"--dry-run"])
            command = plan["AIPerf"]
            self.assertEqual(command[command.index("--concurrency")+1],str(c))
            self.assertEqual(command[command.index("--request-count")+1],"128")
            self.assertEqual(command[command.index("--warmup-request-count")+1],"8")
            self.assertEqual(json.loads(command[command.index("--extra-inputs")+1]),
                {"temperature":0,"ignore_eos":True,"cache_salt":common.NAMESPACE})
            self.assertEqual(plan["profile"]["native_concurrency"],min(c,13))
            self.assertEqual(plan["probe_count"],1)
            self.assertEqual(plan["dataset_sha256"],common.DATA_SHA)
            self.assertFalse(plan["service_verified"])
        self.assertFalse((self.root/"perf").exists())

    def test_quality_preparation_uses_batch13_and_original_generation(self):
        plan = self.capture(accuracy_candidate.main,["--recipe",str(self.path),"--root",str(self.root),
            "--output-dir",str(self.root/"quality"),"--dry-run"])
        self.assertEqual(plan["profile"],dict(name="quality100",mode="accuracy",native_concurrency=13,
                                           client_concurrency=13,kv_capacity=532480))
        self.assertEqual(plan["task_config"]["eval_batch_size"],13)
        self.assertEqual(plan["task_config"]["limit"],100)
        self.assertEqual(plan["task_config"]["datasets"],["ifbench"])
        self.assertEqual(plan["task_config"]["generation_config"],{"max_tokens":32768})
        self.assertEqual(plan["quality_status"],"105_complete_100_strict85_loose88_empty1")
        self.assertFalse(plan["service_verified"])
        self.assertFalse((self.root/"quality").exists())

    def launch_fixture(self):
        launch = common.render_service(self.recipe,self.sha,"perf-c14",root=self.root,log_dir=self.root/"service")
        launch.update(state="exec_pending",pid=123,start_ticks=999,process_environment=launch["environment"],
                      asset_verification={"scope":"CPU mock assets"})
        binary = Path(launch["paths"]["binary"])
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"Never executed CPU fixture")
        proc = self.root/"proc/123"
        proc.mkdir(parents=True)
        (proc/"exe").symlink_to(binary)
        (proc/"stat").write_text("123 (CPU fixture) "+" ".join(["S"]+["0"]*18+["999"]))
        (proc/"cmdline").write_bytes(b"\0".join(x.encode() for x in launch["argv"])+b"\0")
        (proc/"environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k,v in launch["environment"].items())+b"\0")
        (proc/"fd").mkdir(); (proc/"fd/7").symlink_to("socket:[12345]")
        (proc/"net").mkdir()
        (proc/"net/tcp").write_text("header\n0: 0100007F:4843 00000000:0000 0A 0:0 00:0 0 0 0 12345\n")
        path = self.root/"launch.json"
        path.write_text(json.dumps(launch))
        return launch,path,proc

    def check_fixture(self,path):
        with mock.patch.object(common,"hash_file",return_value=self.recipe["binary_sha256"]), \
             mock.patch.object(common,"validate_assets",return_value={"scope":"CPU mock assets"}):
            return common.check_service(path,self.recipe,self.sha,"perf-c14","http://127.0.0.1:18499/v1",self.root/"proc")

    def test_live_receipt_accepts_native13_client14(self):
        launch,path,_ = self.launch_fixture()
        self.assertEqual(self.check_fixture(path),launch)

    def test_client_rejects_native14_and_native_sized_kv(self):
        launch,path,_ = self.launch_fixture()
        for key,value in (("native_concurrency",14),("client_concurrency",13),("kv_capacity",113152)):
            changed = copy.deepcopy(launch)
            changed["profile"][key] = value
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"native/KV/profile"):
                self.check_fixture(path)

    def test_client_rejects_actual_argv_drift_and_other_inventory(self):
        launch,path,proc = self.launch_fixture()
        argv = list(launch["argv"])
        argv[argv.index("--max-concurrency")+1] = "14"
        (proc/"cmdline").write_bytes(b"\0".join(x.encode() for x in argv)+b"\0")
        with self.assertRaisesRegex(ValueError,"Actual service argv"):
            self.check_fixture(path)
        changed = copy.deepcopy(launch)
        changed["asset_contract"]["file"] = "assets-full64.json"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"asset inventory"):
            self.check_fixture(path)

    def test_client_rejects_missing_listener_or_reused_pid(self):
        _,path,proc = self.launch_fixture()
        (proc/"net/tcp").write_text("header\n")
        with self.assertRaisesRegex(ValueError,"listening socket"):
            self.check_fixture(path)
        (proc/"stat").write_text("123 (reused) "+" ".join(["S"]+["0"]*18+["1000"]))
        with self.assertRaisesRegex(ValueError,"PID was reused"):
            self.check_fixture(path)

    def test_evaluation_requires_live_service_receipt_before_runtime_work(self):
        for entry,extra in ((perf_candidate,["--concurrency","14"]),(accuracy_candidate,[])):
            with self.assertRaisesRegex(ValueError,"service-launch"):
                entry.main(["--recipe",str(self.path),"--root",str(self.root),
                            "--output-dir",str(self.root/"evaluation")]+extra)
            self.assertFalse((self.root/"evaluation").exists())

    def asset_fixture(self,recipe):
        contract = common.load_asset_contract(recipe)
        paths = {"assets":str(self.root/recipe["recipe_id"]),"model":str(self.root/(recipe["recipe_id"]+".model"))}
        files = {Path(paths["model"]):contract["model"]["bytes"]}
        files.update({Path(paths["assets"])/name/rel:item["bytes"]
            for name,spec in contract["caches"].items() for rel,item in spec["files"].items()})
        for path,size in files.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("wb") as stream: stream.truncate(size)
        return paths

    def test_both_inventory_sets_are_exact_and_cross_inventory_fails(self):
        full,_ = common.load_recipe(PACKAGE/"candidate_recipes/100-k3-full64.json")
        for recipe,count in ((self.recipe,478),(full,512)):
            paths = self.asset_fixture(recipe)
            result = common.validate_assets(recipe,paths)
            self.assertEqual(sum(x["file_count"] for x in result["caches"].values()),count)
            self.assertFalse(result["content_sha_rehashed"])
            other = full if count==478 else self.recipe
            with self.assertRaisesRegex(ValueError,"layer set mismatch"):
                common.validate_assets(other,paths)
            (Path(paths["assets"])/"weights047/weights-L999-output").mkdir()
            with self.assertRaisesRegex(ValueError,"layer set mismatch"):
                common.validate_assets(recipe,paths)

    def test_truncated_asset_fails_before_service_exec(self):
        paths = self.asset_fixture(self.recipe)
        packed = Path(paths["assets"])/"weights047/weights-L1-output/packed.bin"
        packed.write_bytes(b"truncated")
        # Only the binary SHA is mocked; inventory SHA and all metadata checks stay real.
        original_hash = common.hash_file
        def hash_or_binary(path):
            return self.recipe["binary_sha256"] if Path(path).name=="ninfer-serve" else original_hash(path)
        with mock.patch.object(common,"hash_file",side_effect=hash_or_binary):
            with self.assertRaisesRegex(ValueError,"file size mismatch"):
                serve_candidate.main(["--recipe",str(self.path),"--root",str(self.root),"--profile","perf-c14",
                    "--assets",paths["assets"],"--model",paths["model"],"--log-dir",str(self.root/"service")])
        self.assertFalse((self.root/"service").exists())


if __name__ == "__main__":
    unittest.main()
