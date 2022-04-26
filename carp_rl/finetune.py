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

	if config['use_lm_ckpt']:
		model = GPT2HeadWithValueModel.from_pretrained(config['lm_ckpt_path'])
	else:
		model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
	#Freeze all but last attention layer
	gpt_blocks = list(model.transformer.h)[:-config['num_layers_unfrozen']]
	for m in gpt_blocks:
		for p in m.parameters():
			p.requires_grad = False


	model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
	tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])
	carp = load_carp(config["carp_version"], config["carp_config_path"], config["carp_ckpt_path"])
	print(carp.config.labels)

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
		batch['tokens'] = batch['prompt'].apply(lambda x: tokenizer.encode(x, return_tensors="pt").flatten().to(device))
		min_size = min([item.shape[0] for item in batch['tokens']])
		print(f"MIN SIZE: {min_size}")
		if min_size >= 3:
			#print(batch['tokens'].tolist()[0].shape)
			batch_tokens = [item[:min_size] for item in batch['tokens']]
			#print(batch_tokens)
			#print(min_size)
			#print(batch_tokens)
			query_tensors = torch.stack(batch_tokens)


			response_tensors = []
			for i in range(int(config['batch_size']/fbs)):
				response  = respond_to_batch(model, query_tensors[i*fbs:(i+1)*fbs],
											txt_len=config['txt_out_len'])
				response_tensors.append(response)
			response_tensors = torch.cat(response_tensors)
			stories = []
			for i in range(config['batch_size']):
				try:
					story = tokenizer.decode(response_tensors[i, :])
				except(TypeError):
					tensor = response_tensors[i, :]
					print("Caught type error. Response tensor {tensor}")
					story = None
				stories.append(story)

			scores = []
			#TODO: make this batched
			for story in stories:
				if story is None:
					score = -1.0
				else:
					score = scorer([story], [review], carp, mode=config['carp_version'])
				if config['minimize']:
					score *=-1
				scores.append(score)

			score_mean = sum(scores)/len(scores)
			score_mean = score_mean[0][0].item()
			scores = torch.tensor(scores).to(device)
			wandb.log({'reward': score_mean}) if LOG else None
		else:
			print(f"Skipped batch with min_size {min_size}")

			#Run PPO
		stats = ppo_trainer.step(query_tensors, response_tensors, scores)

	model.save_pretrained(config['save_folder']) if config['save_model'] else None
	run.finish() if LOG else None