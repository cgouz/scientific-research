"""
Translator web app (like Google Translate) for the fine-tuned Riva-Translate GGUF model, on llama.cpp.

It starts llama-server with the GGUF file on the GPU or the CPU (config.yaml: device), serves the page in static/,
and a small JSON API that sends the exact training prompt (the same as ttt/riva_multilingual_train/train.py).

Usage:
  python app.py                                                  # config.yaml -> http://<server>:7860
  python app.py --set device=cpu                                 # run the model on the CPU only
  python app.py --set model=/data/experiments/riva_multilingual/riva-translate-multilingual-Q4_K_M.gguf
  python app.py --set model=/data/experiments/riva_multilingual_auto/riva-translate-multilingual-Q8_0.gguf detect=model

API:
  POST /api/translate  {"text": "...", "source": "detect" | "uz" | "ru" | "en", "target": "uz" | "ru" | "en"}
                   ->  {"translation": "...", "source": "uz", "detected": true, "ms": 123}
  GET  /api/info   ->  model, device, languages, ready

Needs only the Python standard library and pyyaml (no torch / numpy), plus a built llama-server.
"""

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                       # repo root, for general/
sys.path.insert(0, str(HERE.parent / "ttt" / "riva_multilingual_train"))   # detect_lang.py
from detect_lang import detect as detect_rules  # noqa: E402
from general.config import load_config  # noqa: E402

LANG_NAMES = {"uz": "Uzbek", "ru": "Russian", "en": "English"}


def resolve(p):
    p = Path(str(p))
    return p if p.is_absolute() else (HERE / p).resolve()


# ---------------------------------------------------------------------------
# Prompt (identical to ttt/riva_multilingual_train/train.py and ttt/riva_test/riva_test.py)
# ---------------------------------------------------------------------------
def build_prompt(text, pair, auto=False):
    """auto=True: the source language is left out of the prompt, so the model has to detect it from the text."""
    s, t = (LANG_NAMES[x] for x in pair.split("-"))
    system = f"You are an expert at translating text to {t}." if auto else \
        f"You are an expert at translating text from {s} to {t}."
    return (f"<s>System\n{system}</s>\n"
            f"<s>User\nWhat is the {t} translation of the sentence: {text.strip()}</s>\n"
            "<s>Assistant\n")


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(line):
    """Sentences of one line. Pieces like "13.", "A.A." or "г." are joined to the next one instead of standing alone."""
    out = []
    for piece in SENTENCE_END.split(line.strip()):
        if out and len(out[-1]) <= 4:
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [p for p in out if p]


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------
class Llama:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None
        self.url = (cfg["llama_url"] or f"http://127.0.0.1:{cfg['llama_port']}").rstrip("/")
        self.log_path = HERE / "logs" / "llama-server.log"

    def command(self):
        c = self.cfg
        cmd = [str(resolve(c["llama_server"])), "-m", str(resolve(c["model"])),
               "--host", "127.0.0.1", "--port", str(c["llama_port"]),
               "-np", str(c["parallel"]), "-c", str(int(c["parallel"]) * int(c["ctx_per_slot"])),
               "--override-kv", "tokenizer.ggml.add_bos_token=bool:false"]   # the prompt already starts with <s>
        if c["device"] == "cpu":
            cmd += ["-ngl", "0", "--device", "none"]
        else:
            cmd += ["-ngl", str(c["gpu_layers"])]
        if str(c["threads"]).lower() != "auto":
            cmd += ["-t", str(c["threads"])]
        return cmd

    def start(self):
        if self.cfg["llama_url"]:
            print(f"Using the running llama-server at {self.url}")
            return self.wait_ready()
        for key, path in (("llama_server", self.cfg["llama_server"]), ("model", self.cfg["model"])):
            if not resolve(path).exists():
                sys.exit(f"{key} not found: {resolve(path)}  (see translate_app/README.md)")
        env = dict(os.environ)
        # gpu: only the chosen GPU; cpu: no GPU visible at all, so nothing is offloaded
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg["gpu_index"]) if self.cfg["device"] == "gpu" else ""
        self.log_path.parent.mkdir(exist_ok=True)
        cmd = self.command()
        print("Starting:", " ".join(cmd))
        print(f"llama-server log: {self.log_path}")
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env)
        atexit.register(self.stop)
        self.wait_ready()

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                tail = self.log_path.read_text(errors="replace").splitlines()[-25:]
                sys.exit("llama-server stopped while loading:\n  " + "\n  ".join(tail))
            if self.ready():
                print(f"Model loaded in {time.time() - t0:.0f} s")
                return
            time.sleep(1)
        sys.exit(f"llama-server not ready after {timeout} s (see {self.log_path})")

    def ready(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def complete(self, prompt):
        body = {"prompt": prompt, "n_predict": int(self.cfg["max_new_tokens"]), "temperature": 0, "top_k": 1,
                "stop": ["</s>", "<s>"], "cache_prompt": True}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["content"].strip()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class Translator:
    def __init__(self, cfg, llama):
        self.cfg, self.llama = cfg, llama
        self.pool = ThreadPoolExecutor(max_workers=int(cfg["parallel"]))

    def translate(self, text, source, target):
        if target not in LANG_NAMES:
            raise ValueError(f"unknown target language: {target}")
        if source not in LANG_NAMES and source != "detect":
            raise ValueError(f"unknown source language: {source}")
        text = text[:int(self.cfg["max_chars"])]
        detected = source == "detect"
        auto = detected and self.cfg["detect"] == "model"
        if detected:
            source = detect_rules(text)            # with detect=model only shown in the UI; the prompt omits it
        if source == target and not auto:
            return {"translation": text, "source": source, "detected": detected}

        pair = f"{source}-{target}"   # with auto the source name is not used in the prompt
        # keep the line structure; translate sentence by sentence, several at once (llama-server slots)
        lines = text.split("\n")
        jobs = [(i, s) for i, line in enumerate(lines)
                for s in (split_sentences(line) if self.cfg["split_sentences"] else [line.strip()]) if s]
        outs = list(self.pool.map(lambda job: self.llama.complete(build_prompt(job[1], pair, auto)), jobs))
        result = [[] for _ in lines]
        for (i, _), t in zip(jobs, outs):
            result[i].append(t)
        return {"translation": "\n".join(" ".join(parts) for parts in result), "source": source, "detected": detected}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def make_handler(cfg, llama, translator):
    page = HERE / "static" / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                data = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/info":
                self.send_json({"model": Path(str(cfg["model"])).name, "device": cfg["device"],
                                "detect": cfg["detect"], "languages": LANG_NAMES, "max_chars": int(cfg["max_chars"]),
                                "ready": llama.ready()})
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/translate":
                return self.send_json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                text = str(req.get("text", ""))
                if not text.strip():
                    return self.send_json({"translation": "", "source": None, "detected": False, "ms": 0})
                t0 = time.time()
                out = translator.translate(text, req.get("source", "detect"), req.get("target", "en"))
                out["ms"] = round((time.time() - t0) * 1000)
                self.send_json(out)
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
            except (urllib.error.URLError, OSError) as e:
                self.send_json({"error": f"llama-server: {e}"}, 503)

        def log_message(self, fmt, *args):  # one short line per request
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Translator web app on llama.cpp.")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config, e.g. device=cpu")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg["device"] not in ("gpu", "cpu"):
        sys.exit(f"device must be gpu or cpu, not {cfg['device']}")
    if cfg["detect"] not in ("rules", "model"):
        sys.exit(f"detect must be rules or model, not {cfg['detect']}")

    llama = Llama(cfg)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))  # runs atexit -> stops llama-server
    llama.start()
    server = ThreadingHTTPServer((cfg["host"], int(cfg["port"])), make_handler(cfg, llama, Translator(cfg, llama)))
    shown = "localhost" if cfg["host"] in ("0.0.0.0", "127.0.0.1") else cfg["host"]
    print(f"\nTranslator ready: http://{shown}:{cfg['port']}   ({cfg['device']}, {Path(str(cfg['model'])).name})")
    print("Stop: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        llama.stop()


if __name__ == "__main__":
    main()
