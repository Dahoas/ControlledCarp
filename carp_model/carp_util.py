# imports
import math

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from happytransformer import HappyGeneration
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GPT2Tokenizer,
                          PegasusForConditionalGeneration, PegasusTokenizer)
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer

#Context window
N_CTX = 512
def tok(string_batch, model):
	for i, _ in enumerate(string_batch):
		if len(string_batch[i]) > N_CTX:
			string_batch[i] = string_batch[i][-N_CTX:]

	return model.encA.tok(string_batch)

def get_batch_tokens(dataset, inds, model):
	batch = [dataset[ind] for ind in inds]
	pass_batch = [pair[0] for pair in batch]
	rev_batch = [pair[1] for pair in batch]

	pass_tokens = tok(pass_batch, model)
	rev_tokens = tok(rev_batch, model)
	pass_masks = pass_tokens['attention_mask']
	rev_masks = rev_tokens['attention_mask']
	pass_tokens = pass_tokens['input_ids']
	rev_tokens = rev_tokens['input_ids']

	return pass_tokens, pass_masks, rev_tokens, rev_masks


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda'
tokenizer_pegasus = PegasusTokenizer.from_pretrained(model_name)
model_pegasus = PegasusForConditionalGeneration.from_pretrained(model_name).half().to(torch_device)
#Paraphrases using peagasus. Used for softening.
def get_response(input_text,num_return_sequences,num_beams):
	batch = tokenizer_pegasus([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
	translated = model_pegasus.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
	tgt_text = tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)
	return tgt_text

#Given passage and review calls carpModel on them to return logits
def get_passrev_logits(passages, reviews, model):
    pass_tokens = tok(passages, model)
    rev_tokens = tok(reviews, model)
    pass_masks = pass_tokens['attention_mask']
    rev_masks = rev_tokens['attention_mask']
    pass_tokens = pass_tokens['input_ids']
    rev_tokens = rev_tokens['input_ids']

    with torch.no_grad():
      logits = model.getLogits([pass_tokens, pass_masks],
                              [rev_tokens, rev_masks]).type(dtype=torch.float32)
    return logits


def report_logits(logits):
    logits /= 2.7441
    print((logits[0]).cpu().tolist())
    conf = logits.softmax(1)

    for i, row in enumerate(conf):
        for j, col in enumerate(row):
            print(str(i) + "-" + str(j) + ": " + str(round(col.item(), 2)))


def compute_softened_logits(passages, reviews1, reviews2, model, pairs=True):
    logits1 = torch.sum(get_passrev_logits(passages, reviews1, model), dim=-1).unsqueeze(0)/float(len(reviews1))
    if pairs:
      logits2 = torch.sum(get_passrev_logits(passages, reviews2, model), dim=-1).unsqueeze(0)/float(len(reviews2))

      return torch.cat([logits1, logits2], dim=-1)
    else:
      return logits1


#Lots of options to play with here that dictate how the paraphrases are generated.
#Future work is needed
#Outermost function called to compute logits/softened logits
def compute_logit(passages, reviews, model, soften=True,
                        top_k=False, k = 3,
                        ret = True, pairs=True):
    #Softens the classifiers by using paraphrasing.
    if soften:
      if pairs:
        review1_paraphrases = list(set(get_response(reviews[0], num_return_sequences=3, num_beams=3) + [reviews[0]]))
        review2_paraphrases = list(set(get_response(reviews[1], num_return_sequences=3, num_beams=3) + [reviews[1]]))
        print(review1_paraphrases)
        print(review2_paraphrases)

        review1_contextual = list(map(lambda x: "[quote] " + x, review1_paraphrases))
        review2_contextual = list(map(lambda x: "[quote] " + x, review2_paraphrases))


        softened_logits = compute_softened_logits(passages, review1_contextual + review1_paraphrases, review2_contextual + review2_paraphrases, model)
        report_logits(softened_logits)
        if ret: return softened_logits
      else:
        review_paraphrases = list(set(get_response(reviews, num_return_sequences=3, num_beams=3) + [reviews]))
        print(review_paraphrases)

        review_contextual = list(map(lambda x: "[quote] " + x, review_paraphrases))
        softened_logits = compute_softened_logits(passages, review_contextual + review_paraphrases, None, model, pairs=False)

        #softened_logits = (softened_logits/2.7441)
        print(softened_logits.squeeze().cpu().tolist())

        if ret: return softened_logits

