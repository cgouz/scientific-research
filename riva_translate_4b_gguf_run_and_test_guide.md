# Riva Translate 4B GGUF — Run, CUDA Test, and Prompt Guide

This guide continues from `riva_translate_4b_to_gguf_guide.md`.

At this point the converted model already exists:

```text
/data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

Confirmed size:

```text
7.8G
```

## 1. Activate the GGUF environment

```bash
source /workspace/gguf-venv/bin/activate
unset PIP_CONSTRAINT
unset LD_LIBRARY_PATH
```

Then:

```bash
cd /workspace/scientific-research/ttt/third_party/llama.cpp
```

## 2. Check NVIDIA GPU and CUDA

```bash
nvcc --version
nvidia-smi
```

## 3. Build llama.cpp with CUDA

```bash
cmake -B build   -DGGML_CUDA=ON   -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build --config Release -j$(nproc)
```

Check binaries:

```bash
ls build/bin
```

Important binaries:

```text
llama-cli
llama-server
llama-quantize
```

## 4. First GPU test

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -p "Translate this into Russian: Salom, bugun ob-havo juda yaxshi."   -n 100
```

`-ngl 999` offloads as many layers as possible to the NVIDIA GPU.

The model successfully produced Russian output, which confirmed:

- GGUF loaded successfully
- CUDA worked
- GPU inference worked
- model weights were valid

## 5. Interactive repetition problem

In interactive mode, early translations worked:

```text
nima gap
→ Что происходит

ishdaman
→ Я на работе

maktabdaman
→ Я в школе
```

After several turns the model started repeating:

```text
Я в отпуске
```

for unrelated inputs.

The likely reason was accumulated conversation context.

For translation testing, use single-turn mode:

```text
-st
```

or:

```text
--single-turn
```

Example:

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   -p "Translate this into Russian: Men maktabdaman."   -n 64
```

`--temp 0` makes decoding deterministic, which is preferable for translation tests.

## 6. Long-text test

A longer Uzbek paragraph was tested. The model translated most of it correctly, but one semantic error was noticed:

```text
bir piyola choy ichdim
```

became:

```text
я выпил чашку кофе
```

instead of:

```text
я выпил чашку чая
```

This showed that the GGUF worked, but the inference prompt needed to match the training prompt more closely.

## 7. EOS diagnostic

The model sometimes stopped before finishing the whole translation.

For diagnosis only, this was tested:

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   --ignore-eos   -n 512   -p "Translate this into Russian: ..."
```

With `--ignore-eos`, the model continued and completed the translation, but after the real translation ended it generated garbage/repetitive text such as:

```text
~ ~ ~ (Аплодисменты)
```

Conclusion:

- the model does know where to stop
- EOS is meaningful
- `--ignore-eos` is only useful for debugging
- do not use `--ignore-eos` for normal inference

## 8. Find the exact training prompt

The training code was checked with:

```bash
cd /workspace/scientific-research/ttt
```

```bash
grep -nE   'apply_chat_template|system|user|assistant|source|target|pair|prompt|messages'   riva_train/train.py
```

The training script revealed the exact prompt structure:

```text
<s>User
What is the {target_language} translation of the sentence: {source}</s>
<s>Assistant
{target}</s>
```

The test script also states that it uses the same prompt as training.

This was the key discovery.

## 9. Correct inference prompt

The model was not fine-tuned primarily on:

```text
Translate this into Russian: ...
```

It was trained on:

```text
<s>User
What is the Russian translation of the sentence: {source}</s>
<s>Assistant
```

For Uzbek → Russian:

```text
<s>User
What is the Russian translation of the sentence: Bugun havo juda yaxshi.</s>
<s>Assistant
```

For Russian → Uzbek:

```text
<s>User
What is the Uzbek translation of the sentence: Сегодня очень хорошая погода.</s>
<s>Assistant
```

## 10. Correct Uzbek → Russian test

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   -n 512   -p $'<s>User\nWhat is the Russian translation of the sentence: Bugun ertalab odatdagidan ancha erta uyg‘ondim. Tashqariga chiqqanimda havo salqin, osmon esa juda tiniq edi. Ishga ketishdan oldin bir piyola choy ichdim va kun davomida bajarishim kerak bo‘lgan ishlarni rejalashtirdim. Hozir yangi loyiha ustida ishlayapman. Bu loyiha o‘zbek va rus tillari o‘rtasida nutqni avtomatik tarjima qilishga mo‘ljallangan.</s>\n<s>Assistant\n'
```

The `$'...'` Bash syntax converts `\n` into real newline characters.

## 11. Correct Russian → Uzbek test

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   -n 256   -p $'<s>User\nWhat is the Uzbek translation of the sentence: Сегодня очень хорошая погода.</s>\n<s>Assistant\n'
```

## 12. Why exact prompt matching matters

During fine-tuning, the model repeatedly saw:

```text
<s>User
What is the Russian translation of the sentence: ...
</s>
<s>Assistant
...
</s>
```

So for reliable evaluation, inference should use the same structure as:

```text
riva_train/train.py
```

and:

```text
riva_test/riva_test.py
```

## 13. Training directions

The training config includes both directions:

```yaml
pairs: [uz-ru, ru-uz]
```

So the same GGUF model supports:

```text
Uzbek → Russian
Russian → Uzbek
```

The direction is expressed through the target-language phrase in the prompt:

```text
What is the Russian translation of the sentence: ...
```

or:

```text
What is the Uzbek translation of the sentence: ...
```

## 14. Current status

Working pipeline:

```text
Fine-tuned Riva Translate
        ↓
Hugging Face safetensors
        ↓
GGUF conversion
        ↓
F16 GGUF
        ↓
llama.cpp CUDA build
        ↓
NVIDIA GPU inference
        ↓
Uzbek ↔ Russian translation
```

Confirmed model:

```text
/data/experiments/riva/riva-translate-uz-ru-f16.gguf
```

Confirmed:

- GGUF conversion works
- tokenizer works
- llama.cpp loads the model
- CUDA inference works
- Uzbek → Russian works
- Russian → Uzbek is supported by training
- exact fine-tuning prompt format is known

## 15. Recommended next steps

```text
F16 GGUF
   ↓
test exact training prompt
   ↓
compare GGUF output with Hugging Face output
   ↓
verify uz → ru quality
   ↓
verify ru → uz quality
   ↓
quantize
   ├── Q8_0
   ├── Q6_K
   └── Q4_K_M
   ↓
compare
   ├── model size
   ├── VRAM
   ├── speed
   └── translation quality
```

## Quick command summary

Activate:

```bash
source /workspace/gguf-venv/bin/activate
unset PIP_CONSTRAINT
unset LD_LIBRARY_PATH
```

Go to llama.cpp:

```bash
cd /workspace/scientific-research/ttt/third_party/llama.cpp
```

Uzbek → Russian:

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   -n 256   -p $'<s>User\nWhat is the Russian translation of the sentence: Men har kuni ertalab kitob o‘qiyman.</s>\n<s>Assistant\n'
```

Russian → Uzbek:

```bash
./build/bin/llama-cli   -m /data/experiments/riva/riva-translate-uz-ru-f16.gguf   -ngl 999   -st   --temp 0   -n 256   -p $'<s>User\nWhat is the Uzbek translation of the sentence: Я каждое утро читаю книгу.</s>\n<s>Assistant\n'
```
