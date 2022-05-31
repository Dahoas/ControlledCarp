import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/coop/pseudo_coop_short.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

#reviews = ['characters laughing or finding things funny', 'imagery/descriptions', 'characters asking questions', 'praying/religion/church', 'character internal monologues and thoughts', 'crimes', 'fighting']
reviews = ['characters laughing or finding things funny', 'imagery/descriptions', 'fighting']
use_pretrained_lm = [True, False]
for review in reviews:
	for indicator in use_pretrained_lm:
		config['review'] = review
		config['use_lm_ckpt'] = indicator
		review_paraphrase = review.split('/')[0]
		config['save_folder'] = f'ckpts/{review_paraphrase}_{indicator}_coop_model'
		run(config)