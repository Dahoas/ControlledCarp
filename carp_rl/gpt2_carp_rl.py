# imports
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from transformers import GPT2Tokenizer, AutoConfig, AutoModelForCausalLM, GPTNeoForCausalLM
from transformers import AutoModel
import wandb
from carp_model.carp_util import compute_logit
from carp_model.carp_model import ContrastiveModel, TextEncoder
from util.utils import Debugger, get_model_path
import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataset.prompt_generation import load_prompts

from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
import matplotlib.pyplot as plt


Debugger.start()

config = {
    "lm_name": "lvwerra/gpt2-imdb",
    "ref_lm_name": "lvwerra/gpt2-imdb",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "tk_name": "gpt2",
    "steps": 17000,
    "batch_size": 64,
    "forward_batch_size": 64,
    "ppo_epochs": 4,
    "txt_in_len": 15,
    "txt_out_len": 45,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1,
}

model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
carp = ContrastiveModel(TextEncoder(), TextEncoder())
ckpt_path = get_model_path("CARP_L.pt")
carp.load_state_dict(torch.load(ckpt_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model_ref.to(device)
carp.to(device)

# encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt")
query_tensor = query_tensor.to(device)

# get model response
#seems to be model agnostic(as long as causal/autoregressive model)
response_tensor  = respond_to_batch(model, query_tensor)
response_txt = tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
review = [['This is too sad.'], ['This is too angry']]
story = [response_txt]
#What is softened version?
reward = compute_logit(story, review, carp, pairs=False)
reward = reward[0]

#load data
prompts = load_prompts()
df_prompts = pd.DataFrame(prompts, columns=['prompt'])

# train model with ppo

# initialize trainer
ppo_trainer = PPOTrainer(model, model_ref, **config)

mean_scores = []

for epoch in tqdm(range(int(np.ceil(config['steps']/config['batch_size'])))):
    torch.cuda.empty_cache()

    #get batch
    batch = df_prompts.sample(config['batch_size'])
    batch['tokens'] = batch['prompt'].apply(lambda x: tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
    query_tensors = torch.stack(batch['tokens'].tolist())

    response_tensors = respond_to_batch(model, query_tensors, txt_len=config['txt_out_len'])
    stories = [tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]

    #tokenize text
    scores = []
    for story in stories:
        score = compute_logit([story], review, carp, pairs=False)
        scores.append(score)

    score_mean = sum(scores)/len(scores)
    score_mean = score_mean[0][0].item()
    mean_scores.append(score_mean)
    #Debugger.write(score_mean)
    scores = torch.tensor(scores).to(device)

    #Run PPO
    stats = ppo_trainer.step(query_tensors, response_tensors, scores)

x = np.arange(len(mean_scores))
y = mean_scores
plt.clf()
plt.plot(x,y)
plt.savefig('Reward Plot')

torch.save(model.state_dict(), "model.pt")