import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/coop/alignment_coop.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)

reviews = ['neutral', 'good', 'evil']
experiments = [{'review': review, 'save_folder': f'ckpts/{review}_coop_model', 'minimize': False} for review in reviews]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)