# train_riva_uzru.py file:
Fine-tune nvidia/Riva-Translate-4B-Instruct-v2 on Uzbek -> Russian.
ONE self-contained file: every setting is in CONFIG below. No config file, no other scripts needed.

Prompt (identical to riva_test/riva_test.py, so before/after scores are comparable):
    <s>System\nYou are an expert at translating text from Uzbek to Russian.</s>\n
    <s>User\nWhat is the Russian translation of the sentence: {uzbek}</s>\n
    <s>Assistant\n{russian}</s>
The loss is computed ONLY on the Russian translation.

GPU features: bf16 + TF32, fused AdamW, SDPA, auto batch size (probes the GPU with the longest
examples), length-grouped batches, multi-GPU via torchrun, gradient checkpointing.

Usage (paths in CONFIG are relative to THIS file's folder):

- full training:
```sh
python train_riva_uzru.py                                
```

- quick test
```sh
python train_riva_uzru.py --set train.max_steps=200 train.eval_steps=100 data.test_samples=200
```

- change any setting for one run
```sh  
python train_riva_uzru.py --set method=lora train.lr=1e-4          
```

- continue after a crash
```sh
  python train_riva_uzru.py --resume                                 
```  

``` sh
torchrun --nproc_per_node=gpu train_riva_uzru.py                   
```

Output (CONFIG["output"]["dir"], default outputs/riva-uz-ru/):
  final/     full model (method full) or merged model (method lora)  -> riva_test/riva_test.py --model
  adapter/   the LoRA adapter (method lora)
  test_results.json, test_predictions.csv, config_used.json

Requires: torch transformers datasets accelerate sacrebleu  (+ peft only for method=lora)
