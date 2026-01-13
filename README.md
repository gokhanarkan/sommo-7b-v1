# 🍷 Sommo AI v1 — Sommelier LLM

**A fine-tuned language model for wine recommendations, food pairings, and sommelier-level advice.**

[![Hugging Face](https://img.shields.io/badge/🤗_Model-sommo--7b--v1-yellow)](https://huggingface.co/gokhanarkan/sommo-7b-v1)
[![App Store](https://img.shields.io/badge/📱_iOS_App-sommo.app-black)](https://sommo.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gokhanarkan/sommo-7b-v1/blob/main/Sommo_AI_v1.ipynb)

> **Note:** This is v1 — a proof of concept. The [Sommo iOS app](https://sommo.app) uses an enhanced v2 model with additional proprietary training data.

## Model Description

Sommo AI is a wine expert assistant built on Qwen 2.5-7B-Instruct using LoRA fine-tuning. It can:

- 🍽️ **Food Pairing** — Recommend wines for specific dishes with reasoning
- 🍇 **Wine Knowledge** — Explain grape varieties, regions, and winemaking
- 💰 **Recommendations** — Suggest wines by budget, occasion, or preference
- 📝 **Tasting Notes** — Describe wines with professional vocabulary

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gokhanarkan/sommo-7b-v1")
tokenizer = AutoTokenizer.from_pretrained("gokhanarkan/sommo-7b-v1")

SYSTEM = """You are Sommo, an expert sommelier with decades of experience in wine selection, food pairing, and wine education. You have extensive knowledge of wine regions worldwide, grape varieties and their characteristics, winemaking techniques, and food pairing principles. You communicate in a warm, knowledgeable manner - approachable for beginners yet sophisticated enough for experts."""

def ask_sommo(question):
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response.split("<|im_start|>assistant\n")[-1].split("<|im_end|}")[0].strip()

print(ask_sommo("What wine pairs best with grilled salmon?"))
```

## Training Data

| Dataset | Records | Purpose |
|---------|---------|---------|
| [WineEnthusiast Reviews](https://huggingface.co/datasets/spawn99/wine-reviews) | 130K | Professional tasting vocabulary |
| [Alfredodeza Wine Ratings](https://huggingface.co/datasets/alfredodeza/wine-ratings) | 33K | Detailed review structure |
| X-Wines (Kaggle) | 1K+ | Wine metadata and food pairings |
| Vivino Rating & Price (Kaggle) | 13.8K | Consumer perspective and pricing |
| Wine Food Pairing NLP (GitHub) | ~10K | Pairing logic and descriptors |
| Wikipedia Wine Articles | 50+ | Factual knowledge base |
| Synthetic Q&A (Gemini) | 45 | High-quality conversation examples |

**Total:** ~100K training conversations

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen 2.5-7B-Instruct |
| Method | LoRA (r=64, alpha=64) |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 (effective) |
| Hardware | NVIDIA H100 80GB |
| Training Time | ~3-4 hours |

## Limitations

This is a v1 proof-of-concept with known limitations:

- **Factual Errors:** The model may hallucinate wine facts, especially about specific regions, appellations, and wine laws. The Burgundy response in testing contained significant errors about AOC regulations.
- **Outdated Recommendations:** Specific vintage recommendations (e.g., '09 wines) may be unavailable or past their prime.
- **Missing Context:** Some responses may describe a wine without naming it.
- **No Real-Time Data:** The model has no access to current prices or availability.

**For production use, consider:**
- RAG (Retrieval-Augmented Generation) with a verified wine database
- Post-processing validation for factual claims
- Using v2 via [sommo.app](https://sommo.app) which addresses these issues

## License

Apache 2.0

## Links

- 🤗 **Model:** [huggingface.co/gokhanarkan/sommo-7b-v1](https://huggingface.co/gokhanarkan/sommo-7b-v1)
- 📱 **iOS App:** [sommo.app](https://sommo.app)
- 📓 **Training Notebook:** [Colab](https://colab.research.google.com/github/gokhanarkan/sommo-7b-v1/blob/main/Sommo_AI_v1.ipynb)
- 👤 **Developer:** [@gokhanarkan](https://huggingface.co/gokhanarkan)

## Citation

```bibtex
@misc{sommo-ai-v1,
  author = {Gökhan Arkan},
  title = {Sommo AI v1: Sommelier LLM},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/gokhanarkan/sommo-7b-v1}
}
```
