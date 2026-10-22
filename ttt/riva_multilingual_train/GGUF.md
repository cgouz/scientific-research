# Multilingual Riva model → GGUF

Converts the fine-tuned multilingual model (`/data/experiments/riva_multilingual/final`, uz / ru / en in all 6 directions)
to GGUF with `llama.cpp`, quantizes it, and tests it with **exactly the training prompt**.

It reuses what the uz↔ru conversion already set up on the server ([../../riva_translate_4b_to_gguf_guide.md](../../riva_translate_4b_to_gguf_guide.md)):
the `/workspace/gguf-venv` environment and the CUDA build of `ttt/third_party/llama.cpp`. If one of them is missing,
follow sections 4–7 of that guide (venv) or section 3 of [../../riva_translate_4b_gguf_run_and_test_guide.md](../../riva_translate_4b_gguf_run_and_test_guide.md) (build) first.

Two things are different from the old guides, and both matter for quality:
- **The prompt has a System line.** Training used `<s>System\nYou are an expert at translating text from X to Y.</s>\n<s>User\n...`.
  The old guide's prompt left it out. For the multilingual model the System line is the only place that names the
  *source* language, so keep it.
- **Use `llama-completion -no-cnv`, not `llama-cli`.** In this llama.cpp version `llama-cli` is a chat client: it wraps
  every prompt in the GGUF's chat template, so the model never sees the training format. `llama-completion -no-cnv`
  sends the prompt as is.

Paths used below:
```sh
MODEL=/data/experiments/riva_multilingual/final
OUT=/data/experiments/riva_multilingual
LLAMA=/workspace/scientific-research/ttt/third_party/llama.cpp
```

## 1. Shell setup (every new shell)

```sh
source /workspace/gguf-venv/bin/activate
unset PIP_CONSTRAINT
cd /workspace/scientific-research/ttt/third_party/llama.cpp
env -u LD_LIBRARY_PATH python convert_hf_to_gguf.py --help | head -3   # must print the usage, no error
ls build/bin/llama-completion build/bin/llama-quantize build/bin/llama-server
```

`LD_LIBRARY_PATH` is removed **only for the Python converter** (`env -u LD_LIBRARY_PATH python ...`): it points the
converter's CPU torch at the container's NVIDIA torch libraries. Keep it for the llama.cpp binaries: in the NVIDIA
container it also points to the CUDA libraries they need. If you already ran `unset LD_LIBRARY_PATH` in this shell,
open a new shell for building and running llama.cpp.

### Build llama.cpp with CUDA (once, if `build/bin` is missing)

In a fresh shell (no `unset LD_LIBRARY_PATH`), ~5–10 min:
```sh
cd /workspace/scientific-research/ttt/third_party/llama.cpp
nvcc --version || export PATH=/usr/local/cuda/bin:$PATH     # nvcc must be found
nvidia-smi --query-gpu=name,compute_cap --format=csv         # B200: 10.0 -> architecture 100

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100 \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j$(nproc) --target llama-completion llama-quantize llama-server llama-cli
ls build/bin
```
- `CMAKE_CUDA_ARCHITECTURES=100` compiles only for the B200, which is much faster than all architectures. On another GPU
  use its `compute_cap` without the dot (H100: `90`, A100: `80`), or leave the option out.
- If cmake stops at OpenSSL, add `-DLLAMA_OPENSSL=OFF` (only needed for downloading models over HTTPS).
- Check that CUDA is used: `./build/bin/llama-completion --list-devices` should list the B200.

## 2. Check the model

```sh
ls -lh $MODEL                   # config.json, model.safetensors (7.8G), tokenizer.json, tokenizer_config.json, chat_template.jinja
python -c "import json; c = json.load(open('$MODEL/config.json')); print(c['architectures'], c.get('tie_word_embeddings'), c.get('dtype'))"
```
Expected: `['MistralForCausalLM'] True bfloat16`.

`tie_word_embeddings: True` also means the end-of-training warning `missing keys ... ['lm_head.weight']` was harmless:
the output layer is the embedding matrix, so it isn't stored separately.

## 3. Fix the tokenizer class (once)

The converter can't load `"tokenizer_class": "TokenizersBackend"` (written by the new transformers).
Check, back up, and change it to `PreTrainedTokenizerFast`, the same fix as for the uz↔ru model:

```sh
grep tokenizer_class $MODEL/tokenizer_config.json
cp -n $MODEL/tokenizer_config.json $MODEL/tokenizer_config.json.bak      # -n: never overwrite an existing backup
python - <<PY
import json
p = "$MODEL/tokenizer_config.json"
cfg = json.load(open(p, encoding="utf-8"))
print("old:", cfg.get("tokenizer_class"))
cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("new:", cfg["tokenizer_class"])
PY
env -u LD_LIBRARY_PATH python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$MODEL', local_files_only=True)
print(type(t).__name__, len(t), t.bos_token_id, t.eos_token_id, t.encode('Salom dunyo'))"
```
Expected: `PreTrainedTokenizerFast 131072 1 2 [1, ...]`.

If `riva_test.py` (training env, newer transformers) can't load the tokenizer after this change, restore the original
with `cp $MODEL/tokenizer_config.json.bak $MODEL/tokenizer_config.json` before HF testing, and redo this step before converting.

## 4. Convert to F16 GGUF

```sh
df -h /data                     # need ~25 GB free for F16 + Q8_0 + Q4_K_M

env -u LD_LIBRARY_PATH python convert_hf_to_gguf.py $MODEL --outfile $OUT/riva-translate-multilingual-f16.gguf --outtype f16 --dry-run
env -u LD_LIBRARY_PATH python convert_hf_to_gguf.py $MODEL --outfile $OUT/riva-translate-multilingual-f16.gguf --outtype f16
ls -lh $OUT/*.gguf              # ~7.8G
```
The dry run should end with `n_tensors = 308, total_size = 8.4G` and `Dry run, not writing files`.

F16 instead of BF16: the model was saved in bf16; F16 keeps those values with no visible quality loss and is supported
by every llama.cpp backend. If you see NaNs or garbage output with F16 (rare), convert again with `--outtype bf16`.

## 5. Quantize (smaller and faster)

```sh
./build/bin/llama-quantize $OUT/riva-translate-multilingual-f16.gguf $OUT/riva-translate-multilingual-Q8_0.gguf Q8_0
./build/bin/llama-quantize $OUT/riva-translate-multilingual-f16.gguf $OUT/riva-translate-multilingual-Q4_K_M.gguf Q4_K_M
ls -lh $OUT/*.gguf
```

| File | Size (approx.) | Use |
|---|---|---|
| `f16` | 7.8 GB | reference, the same quality as the HF model |
| `Q8_0` | 4.2 GB | almost no quality loss: the default choice for a GPU server |
| `Q4_K_M` | 2.5 GB | CPU / small GPUs; check that the quality loss is acceptable (section 7) |

`Q6_K` (~3.2 GB) is in between, if Q4_K_M loses too much.

## 6. Translate with the exact training prompt

Paste this helper into the shell (bash). It builds the training prompt for any of the 6 directions:

```sh
riva() {  # riva <src> <tgt> "<text>" [model.gguf]     e.g. riva uz en "Salom, qalaysiz?"
  declare -A L=([uz]=Uzbek [ru]=Russian [en]=English)
  local m=${4:-$OUT/riva-translate-multilingual-Q8_0.gguf}
  local p="<s>System"$'\n'"You are an expert at translating text from ${L[$1]} to ${L[$2]}.</s>"$'\n'
  p+="<s>User"$'\n'"What is the ${L[$2]} translation of the sentence: $3</s>"$'\n'"<s>Assistant"$'\n'
  $LLAMA/build/bin/llama-completion -m "$m" -ngl 999 -no-cnv --temp 0 -n 256 --no-display-prompt \
    --override-kv tokenizer.ggml.add_bos_token=bool:false -p "$p" 2>/dev/null \
    | sed 's/ *\[end of text\]//' | sed '/^$/d'
}
```

- `--override-kv tokenizer.ggml.add_bos_token=bool:false`: the prompt already starts with `<s>`, the same as in training
  (`add_special_tokens=False`). Without it, llama.cpp may add a second `<s>` in front.
- `--temp 0`: greedy decoding, the same as `riva_test.py` (`do_sample=False`).
- Generation stops at `</s>` (the EOS token), so there is no need for `--ignore-eos` or a reverse prompt.
- `2>/dev/null` hides the loading log; remove it when something goes wrong.
- The `sed` part removes llama.cpp's `[end of text]` marker (printed when the model stops at `</s>` by itself) and empty lines,
  so only the translation is printed. A translation without that marker in the raw output hit the `-n 256` limit.

Test every direction:
```sh
riva uz ru "Men har kuni ertalab kitob o'qiyman."
riva ru uz "Я каждое утро читаю книгу."
riva uz en "Bugun ertalab bir piyola choy ichdim."
riva en uz "I drank a cup of tea this morning."
riva en ru "The weather is very nice today."
riva ru en "Сегодня очень хорошая погода."
riva uz ru "Men har kuni ertalab kitob o'qiyman." $OUT/riva-translate-multilingual-f16.gguf   # another file
```

Check once that the prompt is tokenized like in training (a single `<s>` at the start, special tokens as single ids):
```sh
$LLAMA/build/bin/llama-completion -m $OUT/riva-translate-multilingual-Q8_0.gguf -no-cnv -n 1 --verbose-prompt \
  --override-kv tokenizer.ggml.add_bos_token=bool:false -p $'<s>System\nYou are' 2>&1 | grep -A8 "prompt:"
```
In the printed token list, the first token must be `1 -> '<s>'`, once, followed by `System`.

## 7. Check the quality of the GGUF files

Compare a few sentences per direction across `f16`, `Q8_0` and `Q4_K_M`, and against the HF model
(`riva_test.py --model $MODEL`, whose scores you already have):
```sh
for q in f16 Q8_0 Q4_K_M; do
  echo "== $q"; riva uz en "Ishga ketishdan oldin bir piyola choy ichdim va kun davomida bajarishim kerak bo'lgan ishlarni rejalashtirdim." $OUT/riva-translate-multilingual-$q.gguf
done
```
`f16` and `Q8_0` should give the same (or nearly the same) translations as the HF model. If `Q4_K_M` changes the meaning,
use `Q6_K` or `Q8_0` instead.

## 8. Serve it over HTTP (optional)

```sh
$LLAMA/build/bin/llama-server -m $OUT/riva-translate-multilingual-Q8_0.gguf -ngl 999 --host 0.0.0.0 --port 8080 \
  --override-kv tokenizer.ggml.add_bos_token=bool:false
```
Use the raw `/completion` endpoint with the training prompt (not `/v1/chat/completions`, which applies the chat template):
```sh
curl -s localhost:8080/completion -d '{
  "prompt": "<s>System\nYou are an expert at translating text from Uzbek to English.</s>\n<s>User\nWhat is the English translation of the sentence: Salom, qalaysiz?</s>\n<s>Assistant\n",
  "n_predict": 256, "temperature": 0}' | python -c "import json, sys; print(json.load(sys.stdin)['content'])"
```

## 9. Clean up

Once the GGUF files work, the training checkpoints are not needed for inference:
```sh
du -sh $OUT/checkpoint-*        # ~45-50 GB each
rm -rf $OUT/checkpoint-*        # keep final/ (HF model) and the .gguf files
rm -f $OUT/train.pid
```
