# imports
import math

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GPT2Tokenizer,
                          PegasusForConditionalGeneration, PegasusTokenizer)
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from carp.pytorch.model.architectures import *
import torch.nn.functional as F
from carp.configs import CARPConfig

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda'
tokenizer_pegasus = PegasusTokenizer.from_pretrained(model_name)
model_pegasus = PegasusForConditionalGeneration.from_pretrained(model_name).half().to(torch_device)
#Paraphrases using peagasus. Used for softening.
def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer_pegasus(input_text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model_pegasus.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

#Given passage and review calls carpModel on them to return logits
def get_passrev_logits(passages, reviews, model):
    pass_tokens = model.passage_encoder.call_tokenizer(passages).to('cuda')
    rev_tokens = model.review_encoder.call_tokenizer(reviews).to('cuda')
    pass_masks = pass_tokens['attention_mask']
    rev_masks = rev_tokens['attention_mask']
    pass_tokens = pass_tokens['input_ids']
    rev_tokens = rev_tokens['input_ids']
    passage_batch = BatchElement(pass_tokens, pass_masks)
    review_batch = BatchElement(rev_tokens, rev_masks)

    with torch.no_grad():
      pass_encs, rev_encs = model.calculate_embeddings([passage_batch], [review_batch])
      confusion_matrix = model.cosine_sim(pass_encs[0], rev_encs[0])*model.logit_scale.exp()
    return confusion_matrix


def compute_softened_logits(passages, reviews, model):
    logits1 = torch.sum(get_passrev_logits(passages, reviews, model), dim=-1).unsqueeze(0)/float(len(reviews))
    return logits1


#Lots of options to play with here that dictate how the paraphrases are generated.
#Future work is needed
#Outermost function called to compute logits/softened logits
def compute_logit(passages, reviews, model):
    review_paraphrases = list(set(get_response(reviews, num_return_sequences=3, num_beams=3) + reviews))
    review_contextual = list(map(lambda x: "[quote] " + x, review_paraphrases))
    softened_logits = compute_softened_logits(passages, review_contextual + review_paraphrases, model)
    return softened_logits

def convert_coop_review_to_index(reviews):
    converter = {
        'Off-prompt': 0,
        'Grammar Usage': 1,
        'Needs Google': 2,
        'Incoherent': 3,
        'Technical Jargon': 4,
        'Redundant': 5
    }
    indices = []
    for review in reviews:
        index = converter.get(review)
        if index is None:
            raise ValueError(f"ERROR: {review} unsupported CoOp review")
        indices.append(index)
    return indices

def compute_coop_logit(passages, reviews, model):
    pass_tokens = model.passage_encoder.call_tokenizer(passages).to('cuda')
    pass_masks = pass_tokens['attention_mask']
    pass_tokens = pass_tokens['input_ids']
    passage_batch = BatchElement(pass_tokens, pass_masks)

    with torch.no_grad():
        #NOTE: Output shape of rev_encs inconsistent with output shape of pass_encs
        pass_encs, rev_encs = model.calculate_embeddings([passage_batch])
        #Extract desired review
        review_inds = convert_coop_review_to_index(reviews)
        rev_enc = rev_encs[review_inds].reshape((len(reviews),-1))
        confusion_matrix = model.cosine_sim(pass_encs[0], rev_enc)*model.logit_scale.exp()
    return confusion_matrix

def scorer(passages, reviews, model, mode='default'):
    if mode == 'default':
        return compute_logit(passages, reviews, model)
    elif mode == 'coop':
        return compute_coop_logit(passages, reviews, model)
    else:
        raise NotImplementedError(f"{mode} unsupported model type")

def load_carp(model_type, config_path, ckpt_path):
    carp_config = CARPConfig.load_yaml(config_path)
    if model_type == 'default':
        carp = CARP(carp_config.model)
    elif model_type == 'cloob':
        carp = CARPCloob(carp_config.model)
    elif model_type == 'coop':
        carp = CARPCoOp(carp_config.model)
    else:
        raise NotImplemented
    carp.load(ckpt_path)
    return carp
