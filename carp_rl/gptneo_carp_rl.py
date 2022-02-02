# imports
import torch
from transformers import GPT2Tokenizer
from trl.gptneo import GPTNeoHeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from transformers import GPT2Tokenizer, AutoConfig, AutoModelForCausalLM, GPTNeoForCausalLM
from transformers import AutoModel
import wandb
from carp_model.carp_util import scorer, load_carp
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
from carp.pytorch.model.architectures import *
from carp.configs import CARPConfig

from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
import matplotlib.pyplot as plt


Debugger.start()

config = {
    "lm_name": "EleutherAI/gpt-neo-125M",
    "ref_lm_name": "EleutherAI/gpt-neo-125M",
    "tk_name": "EleutherAI/gpt-neo-125M",
    "steps": 25000,
    "batch_size": 64,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "txt_in_len": 14,
    "txt_out_len": 30,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    #KL Divergence target
    "target": .5,
    "horizon":500,
    #Discount factor
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
	#Weight of value function computation in loss.
    "vf_coef":0.1,
}

model = GPTNeoHeadWithValueModel.from_pretrained(config['lm_name'])
model_ref = GPTNeoHeadWithValueModel.from_pretrained(config['ref_lm_name'])
tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
carp_config_path = '/mnt/raid/users/AlexH/control_carp/magiCARP/configs'
carp_config_file = 'carp_l.yml'
carp_config_path = os.path.join(carp_config_path, carp_config_file)
carp_ckpt_path = get_model_path("CARP Roberta L/")
carp = load_carp('default', carp_config_path, carp_ckpt_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model_ref.to(device)
carp.to(device)

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
review = ['This is too cheery.']
#review = ['This is too sad']

#load data
data_file = 'alt_prompts.txt'
prompts = load_prompts(data_file)
df_prompts = pd.DataFrame(prompts, columns=['prompt'])

# train model with ppo

# initialize trainer
ppo_trainer = PPOTrainer(model, model_ref, **config)

mean_scores = []
fbs = config['forward_batch_size']
for epoch in tqdm(range(int(np.ceil(config['steps']/config['batch_size'])))):
    torch.cuda.empty_cache()

    #get batch
    batch = df_prompts.sample(config['batch_size'])
    batch['tokens'] = batch['prompt'].apply(lambda x: tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
    query_tensors = torch.stack(batch['tokens'].tolist())

    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(model, query_tensors[i*fbs:(i+1)*fbs],
                                     txt_len=config['txt_out_len'])
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    stories = [tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]

    #tokenize text
    scores = []
    for story in stories:
        score = scorer([story], review, carp)
        scores.append(score)

    score_mean = sum(scores)/len(scores)
    score_mean = score_mean[0][0].item()
    mean_scores.append(score_mean)
    #Debugger.write(score_mean)
    scores = torch.tensor(scores).to(device)

    #Run PPO
    stats = ppo_trainer.step(query_tensors, response_tensors, scores)
    Debugger.write(stats)

x = np.arange(len(mean_scores))
y = mean_scores
plt.clf()
plt.plot(x,y)
plt.savefig('Reward Plot')

torch.save(model.state_dict(), "model.pt")