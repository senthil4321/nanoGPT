import os
import torch
import tiktoken
import sys

# Set up path for nanoGPT project
correct_nanoGPT_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if correct_nanoGPT_script_dir not in sys.path:
    sys.path.insert(0, correct_nanoGPT_script_dir)

# Now import GPT and GPTConfig
from model import GPT, GPTConfig

# Define the base path to the arithmetic model directory
model_base_path = os.path.join(correct_nanoGPT_script_dir, 'out_arithmetic')

# The ckpt.pt is directly under the out_arithmetic directory
checkpoint_path = os.path.join(model_base_path, 'ckpt.pt')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Attempting to load model from: {checkpoint_path}")
print(f"Using device: {device}")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model configuration from checkpoint
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint['model']
# Filter out unused parameters (e.g., from DDP) if they exist
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Load tokenizer
# The meta.pkl is located in the out_arithmetic folder
meta_path = os.path.join(model_base_path, 'meta.pkl')

import pickle
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Generate sample text
start_string = "PROMPT: Love " # Simple arithmetic prompt with only valid characters
num_samples = 3
max_new_tokens = 100 # Max length for generated text
temperature = 0.1  # Much lower for more deterministic output
top_k = 10  # Much lower to restrict to most likely tokens

print(f"\nGenerating {num_samples} samples starting with: '{start_string}'")

# Encode the start string
start_ids = encode(start_string)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Generate text
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')