import torch

from llm.model import LLM
from llm.config import device
from llm.data import decode, vocab_size


model = LLM().to(device)
print(sum(p.numel() for p in model.parameters())/1e6,  'M parameters')


model.load_state_dict(torch.load("llm/llm_weights_bk.pt", map_location=device))
start_token = torch.randint(low=0,
                            high=vocab_size,
                            size=(1, 1),
                            dtype=torch.long,
                            device=device)

print(f"Starting token is : {start_token[0][0]} => {decode(start_token[0].tolist())}\n")

# To generate a full sequence at once:
# print(decode(model.generate_once(start_token, max_new_tokens=200)[0].tolist()), '\n')

# To generate token by token:
for token in model.generate(start_token, max_new_tokens=5000):
    print(decode(token[0].tolist()), end='', flush=True)
else:
    print()
