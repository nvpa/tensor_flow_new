from sentence_tgransformer import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnetbase-v2")
vector = model.encode("BestMovie Forever!")

vector.shape

#result (768,0)
#The number of values, or the dimensions, of the embedding vector depends
#on the underlying embedding model. Let’s explore that for our model:
#This sentence is now encoded in this one vector with a dimension of 768
#numerical value


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
# Test the model with a prompt formatted for Phi-3
prompt = "<|user|>\nCreate ajoke.<|end|>\n<|assistant|>"

print("Generating response ...")
try:
    result = generate_text(prompt)
    print(result[0]['generated_text'])
except Exception as e:
    print(f"An error occurred: {e}")
    

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
output = generator(prompt)
print(output[0]['generated_text'])

sing device: cuda
Loading model: microsoft/Phi-3-mini-4k-instruct...
Loading weights: 100%
 195/195 [00:28<00:00,  7.22it/s, Materializing param=model.norm.weight]
Both `max_new_tokens` (=500) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
Model and tokenizer loaded successfully.
Pipeline created successfully.
Generating response ...
Both `max_new_tokens` (=256) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
 Why don't scientists trust atoms?

Because they make up everything!
Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. Offer to help her get her garden back in shape. 
