"""Compile/run actual native schema and policy from the reproducibly patched source."""
from pathlib import Path
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVE = Path(os.environ.get(
    "INFERX_NINFER_SOURCE_ARCHIVE",
    ROOT / "artifacts/ninfer-performance/source-closure.tar.gz",
)).expanduser()


@unittest.skipUnless(
    SOURCE_ARCHIVE.is_file(),
    "native source archive unavailable; acquire the pinned NInfer source closure "
    "(including xgrammar) and set INFERX_NINFER_SOURCE_ARCHIVE=/path/to/source-closure.tar.gz",
)
class NInferCompletionsContract(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "native C++ compiler unavailable")
    def test_native_completions_contract(self):
        with tempfile.TemporaryDirectory(prefix="ninfer-completions-") as tmp:
            staging = Path(tmp)
            with tarfile.open(SOURCE_ARCHIVE) as archive:
                archive.extractall(staging, filter="data")
            source = staging / "ninfer-4090"
            subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(ROOT / "patches/ninfer-v2-completions.patch")], cwd=source, check=True, capture_output=True, text=True)
            binary = staging / "contract"
            command = ["g++", "-std=c++20", "-O0", "-Wall", "-Wextra", "-I", str(source / "include"), "-I", str(source / "src"), "-I", str(source / "third_party"), str(ROOT / "tests/native/ninfer_completions_contract.cpp"), str(source / "src/serve/openai_schema.cpp"), str(source / "src/serve/translate.cpp"), "-o", str(binary)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("contracts passed", result.stdout)

    @unittest.skipUnless(shutil.which("g++"), "native C++ compiler unavailable")
    def test_real_frontend_contract(self):
        with tempfile.TemporaryDirectory(prefix="ninfer-frontend-") as tmp:
            staging = Path(tmp)
            with tarfile.open(SOURCE_ARCHIVE) as archive:
                archive.extractall(staging, filter="data")
            source = staging / "ninfer-4090"
            subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(ROOT / "patches/ninfer-v2-completions.patch")], cwd=source, check=True, capture_output=True, text=True)
            grammar = staging / "xgrammar_source-src"
            sources = [ROOT / "tests/native/ninfer_completions_frontend.cpp",
                       source / "src/targets/qwen3_6/impl/frontend/frontend.cpp",
                       source / "src/targets/qwen3_6/impl/frontend/tokenizer.cpp",
                       source / "src/targets/qwen3_6/impl/frontend/chat_template.cpp",
                       source / "src/targets/qwen3_6/impl/frontend/processor.cpp",
                       source / "src/targets/qwen3_6/impl/frontend/digest.cpp",
                       source / "src/text/unicode.cpp",
                       source / "third_party/utf8proc/utf8proc.c"]
            sources += sorted(path for path in (grammar / "cpp").rglob("*.cc") if "tvm_ffi" not in path.parts)
            includes = [source / "include", source / "src", source / "third_party",
                        source / "third_party/utf8proc", source / "src/targets/qwen3_6/export",
                        grammar / "include", grammar / "3rdparty/dlpack/include", grammar / "3rdparty/picojson"]
            common = ["-O0", "-ffunction-sections", "-fdata-sections", "-pthread",
                      "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-DXGRAMMAR_ENABLE_INTERNAL_CHECK=0", "-DUTF8PROC_STATIC"]
            for path in includes:
                common += ["-I", str(path)]
            def compile_one(item):
                index, path = item
                obj = staging / f"object-{index}.o"
                language = ["-x", "c", "-std=c11"] if path.suffix == ".c" else ["-std=c++20"]
                result = subprocess.run(["g++", *language, *common, "-c", str(path), "-o", str(obj)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return obj
            with ThreadPoolExecutor(max_workers=4) as pool:
                objects = list(pool.map(compile_one, enumerate(sources)))
            binary = staging / "frontend-contract"
            result = subprocess.run(["g++", "-pthread", "-Wl,--gc-sections", *map(str, objects), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary), str(source / "tests/fixtures/frontend/thinking_toggle_chat_template.jinja")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("contracts passed", result.stdout)
