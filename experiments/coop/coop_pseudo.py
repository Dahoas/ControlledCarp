import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/coop/pseudo_coop.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)
#to modify:
#save folder
#review
#min vs max

reviews = ['characters laughing or finding things funny', 'imagery/descriptions', 'characters asking questions', 'praying/religion/church', 'accident/bad scenarios', 'character internal monologues and thoughts', 'crimes', 'fighting', 'music', 'family']
experiments = [{'review': review, 'save_folder': f'ckpts/{review}_coop_model', 'minimize': False} for review in reviews]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)