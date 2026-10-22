# translate_app

A small Google Translate-style web page for the fine-tuned Riva-Translate GGUF model (Uzbek, Russian, English in all
directions), running on llama.cpp on the **GPU or the CPU**.

```
browser ──► app.py (page + /api/translate) ──► llama-server (llama.cpp, GGUF on GPU or CPU)
                │ builds the exact training prompt, splits text into sentences,
                │ detects the input language (detect_lang.py)
```

- Translates as you type, "Detect language", swap, copy; keeps line breaks; works on a phone; light and dark theme.
- `app.py` needs only Python and `pyyaml`: no torch, no numpy, so no fights with the training environment.
- It starts and stops `llama-server` itself; the web page has no external dependencies (works offline).

## Requirements

- A built `llama-server` with CUDA: `ttt/third_party/llama.cpp/build/bin/llama-server`
  (see [GGUF.md](../ttt/riva_multilingual_train/GGUF.md), "Build llama.cpp with CUDA"; the CPU mode uses the same binary).
- A GGUF file, by default `/data/experiments/riva_multilingual/riva-translate-multilingual-Q8_0.gguf`.
- `pip install pyyaml` (already in both venvs on the server).

## Run

Use a normal shell, **without** `unset LD_LIBRARY_PATH` (llama-server needs the CUDA libraries):

```sh
cd /workspace/scientific-research/translate_app
python app.py                                    # GPU, Q8_0 -> http://<server>:7860
python app.py --set device=cpu                   # CPU only
python app.py --set model=/data/experiments/riva_multilingual/riva-translate-multilingual-Q4_K_M.gguf
```
It prints `Model loaded in N s` and `Translator ready: http://localhost:7860`. Stop with Ctrl+C (llama-server stops too).

In the background (keeps running after you log out):
```sh
nohup python app.py > logs/app.log 2>&1 &
tail -f logs/app.log                             # llama-server's own log: logs/llama-server.log
pkill -f "translate_app/app.py|python app.py"    # stop
```

### Opening the page

- On the server itself: http://localhost:7860
- From your computer over SSH: `ssh -L 7860:localhost:7860 <user>@<server>`, then open http://localhost:7860
- From JupyterLab with jupyter-server-proxy: `https://<jupyter-address>/proxy/7860/` (with the trailing slash)

## GPU or CPU

| Setting | GPU | CPU |
|---|---|---|
| `device` | `gpu` (default) | `cpu` |
| what it does | `-ngl 999`: all layers on the GPU (`gpu_index`) | `-ngl 0 --device none`, GPUs hidden |
| speed (4B model, rough) | well under a second per sentence | a few seconds per sentence; use `Q4_K_M` and set `threads` to your physical cores |
| memory (rough) | ~5 GB VRAM for Q8_0 | ~3–5 GB RAM (Q4_K_M–Q8_0) |

Middle ground: `--set gpu_layers=20` keeps 20 layers on the GPU and the rest on the CPU (for small GPUs).

## Settings ([config.yaml](config.yaml))

| Key | Default | Meaning |
|---|---|---|
| `model` | `.../riva-translate-multilingual-Q8_0.gguf` | GGUF file (f16, Q8_0, Q4_K_M, or the auto-source model) |
| `device` | `gpu` | `gpu` or `cpu` |
| `gpu_layers`, `gpu_index` | `999`, `0` | layers on the GPU, which GPU |
| `threads` | `auto` | CPU threads |
| `parallel` | `4` | sentences translated at the same time |
| `detect` | `rules` | "Detect language": `rules` = [detect_lang.py](../ttt/riva_multilingual_train/detect_lang.py) picks the language, normal prompt (any model); `model` = prompt without the source language, the model detects it (only the model trained with `config_auto.yaml`) |
| `split_sentences` | `true` | long text is translated sentence by sentence (the model was trained on single sentences) |
| `max_chars` | `5000` | longest input |
| `host`, `port` | `0.0.0.0`, `7860` | page address (`127.0.0.1` = only this machine) |
| `llama_url` | `null` | use an llama-server that is already running instead of starting one |

With the auto-source model:
```sh
python app.py --set model=/data/experiments/riva_multilingual_auto/riva-translate-multilingual-Q8_0.gguf detect=model
```

## API

```sh
curl -s localhost:7860/api/translate -H 'Content-Type: application/json' \
  -d '{"text": "Bugun havo juda yaxshi.", "source": "detect", "target": "en"}'
# {"translation": "The weather is very nice today.", "source": "uz", "detected": true, "ms": 180}
curl -s localhost:7860/api/info
```
`source`: `detect`, `uz`, `ru` or `en`; `target`: `uz`, `ru` or `en`.

## Problems

| Message | Fix |
|---|---|
| `llama_server not found` | build llama.cpp (see Requirements) or set `llama_server` |
| `llama-server stopped while loading` + log | the last lines show why: wrong GGUF path, out of memory, CUDA library missing (don't `unset LD_LIBRARY_PATH`) |
| page says `server not reachable` | `app.py` isn't running, or the port isn't forwarded |
| `Address already in use` | another app on that port: `--set port=7870` (or `llama_port=8082`) |
| translations of long paragraphs are odd | keep `split_sentences: true` |
