# imports
import torch
from transformers import GPT2Tokenizer
from trl.gptneo import GPTNeoHeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from transformers import GPT2Tokenizer, AutoConfig, AutoModelForCausalLM, GPTNeoForCausalLM
from transformers import AutoModel
from happytransformer import HappyGeneration
import wandb

# get models
#For some reason installing happytransformer seems to allow these imports to work
#Also this rl stuff takes up a lotttt of mem. Since we also need ref model.
#Auto model vs automodelcausallm?
model = GPTNeoHeadWithValueModel.from_pretrained("EleutherAI/gpt-neo-125M")
model_ref = GPTNeoHeadWithValueModel.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model_ref.to(device)

#?For some reason observing difference between .to(device) vs .cuda()

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")
query_tensor = query_tensor.to(device)

# get model response
#seems to be model agnostic(as long as causal/autoregressive model)
response_tensor  = respond_to_batch(model, query_tensor)
response_txt = tokenizer.decode(response_tensor[0,:])
print("Response: ", response_txt)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = torch.tensor([1.0]).cuda()

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)