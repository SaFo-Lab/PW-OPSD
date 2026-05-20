"""Merge a PEFT LoRA adapter into the base model and save the merged weights.
Usage: python merge_lora.py BASE_MODEL ADAPTER_DIR OUTPUT_DIR
"""
import sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = sys.argv[1]
adapter = sys.argv[2]
out = sys.argv[3]

print(f"Loading base: {base}")
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
print(f"Loading adapter: {adapter}")
m = PeftModel.from_pretrained(m, adapter)
print("Merging...")
m = m.merge_and_unload()
print(f"Saving to: {out}")
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
print(f"Saving tokenizer to: {out}")
t = AutoTokenizer.from_pretrained(base)
t.save_pretrained(out)
print("Done.")
