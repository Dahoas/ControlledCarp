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
    review_contextual = list(map(lambda x: "[quote] " + x, reviews))
    softened_logits = compute_softened_logits(passages, review_contextual + reviews, model)
    return softened_logits

def rejection_sample(passages, reviews, model):
    THRESH = 1.0
    logits = compute_logit(passages, reviews, model)
    scores = torch.tensor([[0.0 if logit < THRESH else 1.0 for logit in logits]])
    return scores

def scorer(passages, reviews, model, mode='default'):
    if mode == 'default':
        return compute_logit(passages, reviews, model)
    elif mode == 'rejection_sample':
        return rejection_sample(passages, reviews, model)
    else:
        raise NotImplementedError
