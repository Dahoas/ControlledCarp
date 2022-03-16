# imports
import argparse
import math
import os
import re
from hashlib import new
from lib2to3.pgen2 import token
from operator import ne

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.model.architectures import carp_cloob, get_architecture
from dataset.prompt_generation import generate_prompts
from happytransformer import HappyGeneration
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GPT2Tokenizer,
                          PegasusForConditionalGeneration, PegasusTokenizer)
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from util.carp_util import compute_logit, load_carp, scorer


#Currently only supporting one critique, gpt2-large base model
def evaluate_model(model_path, carp_mode, carp_config_path, carp_ckpt_path, critique, data_path, num_eval_examples):
	model_name = os.path.basename(model_path).split('.')[0]
	if model_name == "":
		model_name = model_path.split('/')[:-2]
	print(f"Evaluating {model_name}")

	#Loading random evaluation prompts from dataset
	with open(data_path,'r') as f:
		prompts = f.readlines()
	import random
	random.shuffle(prompts)
	batch = prompts[:num_eval_examples]
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
	batch_token =[tokenizer.encode(x, return_tensors="pt").to('cuda')[0, :14] for x in batch]
	query_txt = [tokenizer.decode(tokenized_text) for tokenized_text in batch_token]
	query_tensors = torch.stack(batch_token).to('cuda')

	#Load models
	base_model = GPT2HeadWithValueModel.from_pretrained('gpt2-large')
	base_model.to('cuda')
	tuned_model = GPT2HeadWithValueModel.from_pretrained(model_path)
	tuned_model.to('cuda')
	#tuned_model.load_state_dict(torch.load(model_path))

	#Generate responses
	tuned_response_tensors = respond_to_batch(tuned_model, query_tensors, txt_len=50)
	tuned_stories = [tokenizer.decode(tuned_response_tensors[i, :]) for i in range(len(tuned_response_tensors))]
	base_response_tensors = respond_to_batch(base_model, query_tensors, txt_len=50)
	base_stories = [tokenizer.decode(base_response_tensors[i, :]) for i in range(len(base_response_tensors))]

	#Load carp
	carp = load_carp(carp_mode, carp_config_path, carp_ckpt_path)
	carp.to('cuda')

	#Score Text
	with open(f'results/{model_name}_results.txt', 'w') as f:
		print("Writing")
		f.write(f"Model Type: {model_name}\n")
		f.write(f"Critique: {critique}\n")
		for prompt, (base, tuned) in zip(query_txt, zip(base_stories, tuned_stories)):
			base_story = prompt + " " + base
			tuned_story = prompt + " " + tuned
			base_score = scorer([base_story], [critique], carp).item()
			tuned_score = scorer([tuned_story], [critique], carp).item()
			f.write(f'Base model: {base_score}: ' + prompt + " " + base + "\n")
			f.write(f'Tuned model: {tuned_score}: ' + prompt + " " + tuned + "\n\n")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", type=str)
	parser.add_argument("--carp_mode", type=str, default="default")
	parser.add_argument("--carp_model_path", type=str, default="/srv/share2/ahavrilla3/ControlledCarp/ckpts/CARP Roberta L/")
	parser.add_argument("--carp_config_path", type=str, default="/srv/share2/ahavrilla3/magiCARP/configs/carp_l.yml")
	parser.add_argument("--critique", type=str)
	parser.add_argument("--data_path", type=str, default="dataset/alt_prompts.txt")
	parser.add_argument("--num_eval_examples", type=int, default=10)

	args = parser.parse_args()
	evaluate_model(args.model_path, args.carp_mode, args.carp_config_path, args.carp_model_path, args.critique,
					args.data_path, args.num_eval_examples)

