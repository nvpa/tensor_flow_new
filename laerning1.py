from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Upgrade transformers to the latest version to fix KeyError: 'type'
!pip install -U transformers

# Load model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load config and remove rope_scaling if it is set to 'default'
config = AutoConfig.from_pretrained(model_id)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    if isinstance(config.rope_scaling, dict) and config.rope_scaling.get("type") == "default":
        config.rope_scaling = None

print(f"Loading model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map=device,
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Model and tokenizer loaded successfully.")
from transformers import pipeline

# Create a pipeline with explicit generation parameters to avoid warnings
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=device
)

# Example usage with explicit parameters
def generate_text(prompt):
    return generator(
        prompt,
        max_new_tokens=500,
        max_length=None, # Explicitly set to None to suppress the warning
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=True
    )

print("Pipeline created successfully.")

prompt = "<|user|>\nCreate ajoke.<|end|>\n<|assistant|>"

print("Generating response ...")
try:
    result = generate_text(prompt)
    print(result[0]['generated_text'])
except Exception as e:
    print(f"An error occurred: {e}")
