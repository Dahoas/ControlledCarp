import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/default/circle_prompt_finetuned.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

reviews = ['This story is action packed.',
		   'This story is so funny.'
]
use_pretrained_lm = [True, False]

for review in reviews:
	for indicator in use_pretrained_lm:
		config['review'] = review
		config['use_lm_ckpt'] = indicator
		config['save_folder'] = f'ckpts/{review}_{indicator}_default_model'
		run(config)