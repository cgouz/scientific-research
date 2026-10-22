# Riva Translate 4B → GGUF Conversion Guide

This guide documents the exact working process used to convert the trained Riva Translate 4B model into GGUF format with `llama.cpp`.

## Environment

Project:

```text
/workspace/scientific-research/ttt
```

`llama.cpp` submodule:

```text
/workspace/scientific-research/ttt/third_party/llama.cpp
```

Trained model:

```text
/data/experiments/riva/final
```

Final GGUF:

```text
/data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

Result:

```text
-rw-r--r-- 1 root 4500110 7.8G Oct 15 04:56 /data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

---

## 1. Check `llama.cpp`

```bash
ls /workspace/scientific-research/ttt/third_party/llama.cpp
```

Make sure this file exists:

```text
convert_hf_to_gguf.py
```

---

## 2. Check trained model files

```bash
ls -lah /data/experiments/riva/final
```

The model contained:

```text
chat_template.jinja
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

The model was about:

```text
7.8G
```

---

## 3. Check model architecture

```bash
cat /data/experiments/riva/final/config.json
```

Important values:

```json
{
  "architectures": [
    "MistralForCausalLM"
  ],
  "dtype": "bfloat16",
  "head_dim": 128,
  "hidden_size": 3072,
  "intermediate_size": 8640,
  "max_position_embeddings": 8192,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 34,
  "num_key_value_heads": 8,
  "tie_word_embeddings": true,
  "vocab_size": 131072
}
```

So the trained model is Mistral-compatible.

---

## 4. Create a separate GGUF Python environment

Do not use the training environment for conversion.

Deactivate the current environment:

```bash
deactivate
```

Create a new environment:

```bash
python3 -m venv /workspace/gguf-venv
```

Activate it:

```bash
source /workspace/gguf-venv/bin/activate
```

Check:

```bash
which python
python --version
```

Expected:

```text
/workspace/gguf-venv/bin/python
Python 3.12.3
```

---

## 5. Remove NVIDIA pip constraint for this shell

The NVIDIA server image had:

```text
PIP_CONSTRAINT=/etc/pip/constraint.txt
```

This forced NVIDIA's PyTorch version and caused dependency conflicts.

Check:

```bash
env | grep -iE 'PIP|CONSTRAINT|TORCH'
```

Disable the constraint only for the current shell:

```bash
unset PIP_CONSTRAINT
```

Check:

```bash
echo $PIP_CONSTRAINT
```

It should print nothing.

---

## 6. Install `llama.cpp` conversion dependencies

Go to `llama.cpp`:

```bash
cd /workspace/scientific-research/ttt/third_party/llama.cpp
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This installed CPU PyTorch:

```text
torch 2.11.0+cpu
```

---

## 7. Fix PyTorch native-library conflict

Initially:

```bash
python convert_hf_to_gguf.py --help
```

failed with:

```text
AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'
```

The reason was that the system NVIDIA PyTorch libraries were being injected through:

```text
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:...
```

Save the old value if desired:

```bash
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
```

Then unset it for the GGUF conversion shell:

```bash
unset LD_LIBRARY_PATH
```

Test PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.__file__)"
```

Working result:

```text
2.11.0+cpu
/workspace/gguf-venv/lib/python3.12/site-packages/torch/__init__.py
```

Then test the converter:

```bash
python convert_hf_to_gguf.py --help
```

It should display the help output successfully.

> Important: CUDA was not removed from the machine. We only prevented system NVIDIA PyTorch libraries from conflicting with the CPU PyTorch used by the converter.

---

## 8. Fix tokenizer metadata

The first dry-run failed because the converter tried several tokenizer paths and eventually failed with:

```text
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

The original tokenizer config was:

```json
{
  "add_prefix_space": false,
  "backend": "tokenizers",
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "</s>",
  "is_local": false,
  "local_files_only": false,
  "model_input_names": [
    "input_ids",
    "attention_mask"
  ],
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "</s>",
  "tokenizer_class": "TokenizersBackend",
  "unk_token": "<unk>"
}
```

The important problem was:

```json
"tokenizer_class": "TokenizersBackend"
```

### Check tokenizer type

The tokenizer was confirmed to be BPE:

```bash
python - <<'PY'
import json

p = "/data/experiments/riva/final/tokenizer.json"

with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

print("model.type =", d.get("model", {}).get("type"))
print("pre_tokenizer =", d.get("pre_tokenizer", {}).get("type"))
PY
```

Result:

```text
model.type = BPE
pre_tokenizer = Sequence
```

So `tokenizer.model` was not required; this model uses `tokenizer.json`.

---

## 9. Backup and fix `tokenizer_config.json`

Create a backup:

```bash
cp /data/experiments/riva/final/tokenizer_config.json \
   /data/experiments/riva/final/tokenizer_config.json.bak
```

Change:

```text
TokenizersBackend
```

to:

```text
PreTrainedTokenizerFast
```

with:

```bash
python - <<'PY'
import json

path = "/data/experiments/riva/final/tokenizer_config.json"

with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

print("old tokenizer_class:", cfg.get("tokenizer_class"))

cfg["tokenizer_class"] = "PreTrainedTokenizerFast"

with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print("new tokenizer_class:", cfg["tokenizer_class"])
PY
```

Expected:

```text
old tokenizer_class: TokenizersBackend
new tokenizer_class: PreTrainedTokenizerFast
```

---

## 10. Test tokenizer

```bash
python - <<'PY'
from transformers import AutoTokenizer

path = "/data/experiments/riva/final"

tok = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True
)

print("class:", tok.__class__.__name__)
print("vocab size:", len(tok))
print("bos:", tok.bos_token, tok.bos_token_id)
print("eos:", tok.eos_token, tok.eos_token_id)
print("pad:", tok.pad_token, tok.pad_token_id)

text = "Salom dunyo"
ids = tok.encode(text)

print("text:", text)
print("ids:", ids)
print("decoded:", tok.decode(ids))
PY
```

Working result:

```text
class: PreTrainedTokenizerFast
vocab size: 131072
bos: <s> 1
eos: </s> 2
pad: </s> 2
text: Salom dunyo
ids: [1, 24683, 1313, 21953, 13504]
decoded: <s>Salom dunyo
```

---

## 11. Run GGUF dry-run

Before writing the real GGUF:

```bash
python convert_hf_to_gguf.py \
    /data/experiments/riva/final \
    --outfile /data/experiments/riva/riva-translate-uz-ru-{ftype}.gguf \
    --outtype f16 \
    --dry-run \
    --verbose
```

Successful result included:

```text
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:/data/experiments/riva/riva-translate-uz-ru-f16.gguf: n_tensors = 308, total_size = 8.4G
INFO:gguf.gguf_writer:Dry run, not writing files
/data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

This confirmed:

- model architecture was supported
- tokenizer loaded correctly
- tensor conversion plan was valid
- output GGUF could be created

---

## 12. Check disk space

```bash
df -h /data
```

At conversion time:

```text
Filesystem                                                                    Size  Used Avail Use% Mounted on
100.126.2.1@o2ib,100.126.2.2@o2ib:100.126.2.3@o2ib,100.126.2.4@o2ib:/ddnaifs  2.8T  517G  2.3T  19% /data
```

There was plenty of free space.

---

## 13. Convert to real F16 GGUF

Run:

```bash
python convert_hf_to_gguf.py \
    /data/experiments/riva/final \
    --outfile /data/experiments/riva/riva-translate-uz-ru-f16.gguf \
    --outtype f16 \
    --verbose
```

Final output file:

```text
/data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

Check:

```bash
ls -lh /data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

Successful result:

```text
-rw-r--r-- 1 root 4500110 7.8G Oct 15 04:56 /data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

---

# Final Working Conversion Pipeline

```text
Trained Riva Translate model
/data/experiments/riva/final
        │
        │ Hugging Face format
        │ model.safetensors
        │ tokenizer.json
        │ config.json
        ▼
llama.cpp
convert_hf_to_gguf.py
        │
        │ F16 conversion
        ▼
riva-translate-uz-ru-f16.gguf
        │
        │ 7.8 GB
        ▼
Ready for llama.cpp loading / testing / quantization
```

---

# Important Notes

## Every new server shell

Because the NVIDIA container defines environment variables globally, when opening a fresh shell for GGUF conversion, use:

```bash
source /workspace/gguf-venv/bin/activate
unset PIP_CONSTRAINT
unset LD_LIBRARY_PATH
```

Then:

```bash
cd /workspace/scientific-research/ttt/third_party/llama.cpp
```

## Do not delete the tokenizer backup

Keep:

```text
/data/experiments/riva/final/tokenizer_config.json.bak
```

It contains the original tokenizer metadata.

## Chat template

The existing `chat_template.jinja` was successfully embedded into GGUF.

However, the language-pair mapping printed during conversion contained many pairs such as:

```text
en-ru
ru-en
en-de
de-en
...
```

but did not visibly contain:

```text
uz-ru
ru-uz
```

Since the model was trained for Uzbek ↔ Russian, the exact inference prompt/chat-template behavior should be tested before considering deployment complete.

Later finding: the chat template is not used for this model. Run it with `llama-completion -no-cnv` and the exact
training prompt (with the System line); see `riva_translate_4b_gguf_run_and_test_guide.md`. The multilingual model
(uz / ru / en) is converted with `ttt/riva_multilingual_train/GGUF.md`.

---

# Next Steps

The F16 GGUF is now ready.

Recommended next pipeline:

```text
F16 GGUF
   ↓
inspect GGUF metadata
   ↓
build llama.cpp with CUDA
   ↓
load and test F16 model
   ↓
test Uzbek → Russian
   ↓
test Russian → Uzbek
   ↓
quantize
   ├── Q8_0
   └── Q4_K_M
   ↓
compare quality / speed / VRAM
```
