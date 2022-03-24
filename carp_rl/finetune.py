import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from util.carp_util import load_carp, scorer
from util.data_utils import load_prompts


def finetune(config):

	LOG = config['LOG']
	run = wandb.init(entity='dahoas', config=config, reinit=True) if LOG else None

	model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
	#Freeze all but last attention layer
	gpt_blocks = list(model.transformer.h)[:-config['num_layers_frozen']]
	for m in gpt_blocks:
		for p in m.parameters():
			p.requires_grad = False


	model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
	tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
	carp = load_carp(config["carp_version"], config["carp_config_path"], config["carp_ckpt_path"])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	model.to(device)
	model_ref.to(device)
	carp.to(device)

	review = config['review']
	#load data
	prompts = load_prompts(config["data_path"])
	df_prompts = pd.DataFrame(prompts, columns=['prompt'])

	# initialize trainer
	ppo_trainer = PPOTrainer(model, model_ref, **config)
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

		scores = []
		#TODO: make this batched
		for story in stories:
			score = scorer([story], [review], carp, mode=config['carp_version'])
			if config['minimize']:
				score *=-1
			scores.append(score)

		score_mean = sum(scores)/len(scores)
		score_mean = score_mean[0][0].item()
		scores = torch.tensor(scores).to(device)
		wandb.log({'reward': score_mean}) if LOG else None

		#Run PPO
		stats = ppo_trainer.step(query_tensors, response_tensors, scores)

	model.save_pretrained(config['save_folder']) if config['save_model'] else None
	run.finish() if LOG else None