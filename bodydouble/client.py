"""Run API-format graphs on a local ComfyUI.

The default transport is comfy-sdk talking to comfy-api-proxy (the Comfy API v2 surface). A raw /prompt transport
exists only as a fallback for mechanics runs; every job writes one line to sdk.log saying which transport ran it.
"""
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

COMFYUI = os.environ.get("BODYDOUBLE_COMFYUI", "http://127.0.0.1:8188")
os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", Path.home() / "ComfyUI"))

import httpx  # noqa: E402
from comfy_sdk import Comfy, ComfyError, JobFailed, Progress  # noqa: E402

# Peak GiB a stack needs beyond what is already resident, used by the admission check (est. on GB10).
STACK_PEAK = {"scail": 45, "klein": 18, "judge": 12, "sam": 6}
OUTPUT_KEYS = ("images", "gifs", "audio", "files", "video")


class RunError(RuntimeError):
    pass


def http_json(url, data=None, timeout=30):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def annotated(out):
    """LoadImage/LoadVideo path for a file ComfyUI saved: 'sub/dir/name.png [output]'."""
    name = f"{out['subfolder']}/{out['filename']}" if out["subfolder"] else out["filename"]
    return f"{name} [{out['type']}]"


def _asset_ref(asset_id):
    payload = asset_id.split(".")[0]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class Client:
    def __init__(self, log_path, transport="sdk"):
        self.log_path = Path(log_path)
        self.transport = transport
        self.sdk = Comfy(timeout=120, client_info="body-double") if transport == "sdk" else None
        self.warm = set()  # stacks loaded since the last /free
        self._info = {}

    def node_info(self, cls):
        if cls not in self._info:
            info = http_json(f"{COMFYUI}/object_info/{cls}")
            if cls not in (info or {}):
                raise RunError(f"ComfyUI has no node {cls}")
            self._info[cls] = info[cls]
        return self._info[cls]

    def check(self, graph):
        """Every node exists and every input key is a real input name (API keys, not display names)."""
        for nid, node in graph.items():
            spec = self.node_info(node["class_type"])["input"]
            names = set(spec.get("required", {})) | set(spec.get("optional", {}))
            for key in node["inputs"]:
                if key.split(".")[0] not in names:
                    raise RunError(f"node {nid} ({node['class_type']}) has no input {key!r}")

    # -- server state -------------------------------------------------------------------------------------------
    def alive(self):
        try:
            http_json(f"{COMFYUI}/system_stats", timeout=10)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def ensure_alive(self):
        """Wait for ComfyUI; after 10 min down, restart it once with the same argv (pre-approved for this box)."""
        down_since = None
        while not self.alive():
            down_since = down_since or time.time()
            if time.time() - down_since > 600:
                self.log({"event": "restart_comfyui"})
                subprocess.Popen([str(COMFYUI_DIR / ".venv/bin/python"), "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                                 cwd=COMFYUI_DIR, stdout=open(COMFYUI_DIR / "user/bodydouble_restart.log", "a"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(120):
                    time.sleep(5)
                    if self.alive():
                        self.warm.clear()
                        return
                raise RunError("ComfyUI did not come back after a restart")
            time.sleep(30)

    def free(self, reason):
        http_json(f"{COMFYUI}/free", {"unload_models": True, "free_memory": True})
        self.warm.clear()
        self.log({"event": "free", "reason": reason})
        time.sleep(5)

    def admit(self, stack):
        """At most two stacks warm; free first if the next stack would make a third or memory is short."""
        if stack in self.warm:
            return
        if len(self.warm) >= 2:
            self.free(f"stack switch {sorted(self.warm)} -> {stack}")
        elif mem_available_gib() < STACK_PEAK.get(stack, 10) and self.warm:
            self.free(f"MemAvailable {mem_available_gib():.0f} GiB < {STACK_PEAK.get(stack)} for {stack}")
        q = http_json(f"{COMFYUI}/queue")
        if q["queue_running"] or q["queue_pending"]:
            self.log({"event": "contended", "running": len(q["queue_running"]), "pending": len(q["queue_pending"])})
        self.warm.add(stack)

    def log(self, rec):
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- running ------------------------------------------------------------------------------------------------
    def run(self, graph, tag, stack, progress=None):
        """Run one graph to completion. Returns [{node, filename, subfolder, type}] in server order."""
        self.check(graph)
        self.ensure_alive()
        self.admit(stack)
        t0 = time.time()
        if self.transport == "sdk":
            try:
                job_id, outs = self._run_sdk(graph, progress)
                transport = "SDK"
            except JobFailed as e:
                self.log({"tag": tag, "transport": "SDK", "status": "failed", "error": str(e.error)[:500], "secs": round(time.time() - t0)})
                raise RunError(f"{tag}: {e.error}") from e
            except (ComfyError, httpx.HTTPError) as e:
                # The proxy or the stream broke. Let anything already queued finish before re-running raw, so a
                # job the proxy did deliver is never doubled.
                self.log({"tag": tag, "transport": "SDK", "status": "transport_error", "error": repr(e)[:300]})
                self._drain()
                job_id, outs = self._run_raw(graph)
                transport = "non-SDK"
        else:
            job_id, outs = self._run_raw(graph)
            transport = "non-SDK"
        self.log({"tag": tag, "job_id": job_id, "prompt_id": job_id, "transport": transport, "status": "success",
                  "sdk": version("comfy-sdk"), "proxy": version("comfy-api-proxy"), "secs": round(time.time() - t0)})
        return outs

    def _run_sdk(self, graph, progress):
        job = self.sdk.submit(self.sdk.workflows.from_json(graph))
        try:
            for ev in job.events():
                if progress and isinstance(ev, Progress):
                    progress(ev)
        except (ComfyError, httpx.HTTPError):
            pass  # the poll below is authoritative
        job.result()
        outs = []
        for o in job.outputs:
            ref = _asset_ref(o.id)
            outs.append({"node": o.node_id, "filename": ref["f"], "subfolder": ref["s"], "type": ref["t"]})
        return job.id, outs

    def _drain(self):
        while True:
            q = http_json(f"{COMFYUI}/queue")
            if not q["queue_running"] and not q["queue_pending"]:
                return
            time.sleep(10)

    def _run_raw(self, graph):
        pid = str(uuid.uuid4())
        try:
            http_json(f"{COMFYUI}/prompt", {"prompt": graph, "prompt_id": pid, "client_id": "body-double"})
        except urllib.error.HTTPError as e:
            raise RunError(e.read().decode()[:2000]) from e
        while True:
            h = http_json(f"{COMFYUI}/history/{pid}")
            if pid in h:
                break
            time.sleep(3)
        entry = h[pid]
        if entry["status"]["status_str"] != "success":
            msgs = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
            raise RunError(json.dumps(msgs)[:2000])
        outs = []
        for node, node_out in entry["outputs"].items():
            for key in OUTPUT_KEYS:
                for it in node_out.get(key, []):
                    if isinstance(it, dict) and "filename" in it:
                        outs.append({"node": node, "filename": it["filename"], "subfolder": it.get("subfolder", ""), "type": it.get("type", "output")})
        return pid, outs
