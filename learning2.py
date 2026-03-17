from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Fix RoPE scaling to avoid known compatibility errors
config = AutoConfig.from_pretrained(model_id)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    if isinstance(config.rope_scaling, dict) and config.rope_scaling.get("type") == "default":
        config.rope_scaling = None

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    config=config,
    torch_dtype="auto",
    trust_remote_code=False
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print("Input IDs created successfully:")
print(input_ids)

# Corrected argument name and variable name typo
generate_output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(generate_output[0]))
