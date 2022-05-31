import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/coop/pseudo_coop_short.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

reviews = ['characters laughing or finding things funny']
lrs = [5.0e-6, 1.24e-5, 5e-5]
vf_coefs = [.12, .25, .5]
layers_unfrozen = [1]
use_lm_ckpts = [False]
kl_targets = [10]
for review in reviews:
	for lr in lrs:
		for kl_target in kl_targets:
			for vf_coef in vf_coefs:
				for layer_unfrozen in layers_unfrozen:
					for use_ckpt in use_lm_ckpts:
						config['review'] = review
						config['lr'] = lr
						config['target'] = kl_target
						config['num_layers_unfrozen'] = layer_unfrozen
						config['use_lm_ckpt'] = use_ckpt
						config['save_folder'] = f'ckpts/humor_{lr}_{kl_target}_{vf_coef}_{layer_unfrozen}_{use_ckpt}_coop_model'
						run(config)