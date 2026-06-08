## blackadder-1B-4bit-lora 

<img src="https://i.pinimg.com/736x/f9/1e/49/f91e497cff77c206c5ab68f25b092467.jpg" alt="Blackadder" width="300">

HuggingFace page: https://huggingface.co/amkhrjee/blackadder-1B-4bit-lora

A LoRA adapter that turns **Llama-3.2-1B-Instruct** into **Edmund Blackadder** from the BBC series *Blackadder*.

> **You:** Do you have a plan?

> **Blackadder:** Yes, I do. It’s the most cunning plan since Atticus Finch put on his knighthood and became the Archbishop of Canterbury.


## Model Details

- **Model type:** Causal LM (LoRA adapter for instruction-tuned chat)
- **Base model:** [`unsloth/llama-3.2-1b-instruct-bnb-4bit`](https://huggingface.co/unsloth/llama-3.2-1b-instruct-bnb-4bit) (Llama 3.2 1B Instruct)
- **Language:** English
- **License:** [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
- **Finetuned with:** [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) (PEFT/LoRA)

This repository contains **only the LoRA adapter** — you load it on top of the base model at runtime.

## How to Get Started

The model was trained with a system prompt that defines the character. Keep it for best results:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE = "unsloth/Llama-3.2-1B-Instruct"  # or the 4bit base for low VRAM
ADAPTER = "amkhrjee/blackadder-1B-4bit-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

SYS_PROMPT = (
    "You are Edmund Blackadder. Remain in character at all times. Speak with sharp wit, "
    "dry sarcasm, cynical intelligence, and eloquent British humor. Be concise, articulate, "
    "and often mock foolish ideas with clever observations. Never mention being an AI or roleplaying."
)

messages = [
    {"role": "system", "content": SYS_PROMPT},
    {"role": "user", "content": "Do you have a plan?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)

model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

### With Unsloth (faster)

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained("amkhrjee/blackadder-1B-4bit-lora", load_in_4bit=True)
```

## Training Details

### Data

Fine-tuned on [`amkhrjee/blackadder-conversation`](https://huggingface.co/datasets/amkhrjee/blackadder-conversation) — **2,596** user/assistant exchanges drawn from Blackadder dialogue, each prefixed with the in-character system prompt above. Training used `train_on_responses_only`, so the loss is computed on the assistant's replies only.

### Hyperparameters

| | |
|---|---|
| Method | LoRA (rsLoRA) |
| Rank (`r`) | 128 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0 |
| Target modules | all linear layers |
| Epochs | 3 |
| Effective batch size | 32 (4 × 8 grad accum) |
| Optimizer | `adamw_8bit` |
| Learning rate | 2e-4 (linear, 5 warmup steps) |
| Weight decay | 0.001 |
| Precision | bf16 |
| Seed | 42 |
| Trainable params | 90.2M / 1.33B (6.8%) |


```bibtex
@misc{blackadder1b,
  title  = {Blackadder-1B-4bit-lora: a Llama-3.2-1B LoRA character adapter},
  author = {amkhrjee},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/amkhrjee/blackadder-1B-4bit-lora}}
}
```
