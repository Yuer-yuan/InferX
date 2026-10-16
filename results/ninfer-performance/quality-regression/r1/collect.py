"""Run the three recorded supplementary cases AFTER the original judge exits.

Usage: python collect.py /absolute/new/output/directory
Reads corpus.json/protocol.json beside this script. No retry, no score inference.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


source = Path(__file__).resolve().parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
for name in ("corpus.json", "protocol.json", "collect.py", "code-merge-tests.py"):
    shutil.copy2(source / name, out / name)
rows = []
for case in json.loads((source / "corpus.json").read_text()):
    folder = out / case["id"]
    folder.mkdir()
    # Match the frozen greedy service and the accuracy client's explicit
    # temperature0, while using the recorded case-specific output budget.
    payload = {"model": "qwen38-27b", "messages": case["messages"],
               "max_tokens": case["max_tokens"], "temperature": 0, "stream": False}
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    save(folder / "request.json", payload)
    row = {"id": case["id"], "request_sha256": hashlib.sha256(encoded).hexdigest(),
           "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        request = urllib.request.Request("http://127.0.0.1:8088/v1/chat/completions",
                                         encoded, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read()
            (folder / "response.raw.json").write_bytes(raw)
            data = json.loads(raw)
            row.update(http_status=response.status, usage=data.get("usage"),
                       finish_reason=data["choices"][0]["finish_reason"],
                       content=data["choices"][0]["message"].get("content"))
            row["transport_complete"] = bool(row["finish_reason"])
    except urllib.error.HTTPError as error:
        (folder / "http-error-body.txt").write_bytes(error.read())
        row.update(transport_complete=False, http_status=error.code, error=str(error))
    except Exception as error:
        row.update(transport_complete=False, error=repr(error))
    row["wall_s"] = time.monotonic() - start
    rows.append(row)
    save(out / "responses.json", rows)
    print(case["id"], row.get("http_status"), row.get("finish_reason"), flush=True)
raise SystemExit(0 if all(r["transport_complete"] for r in rows) else 1)
