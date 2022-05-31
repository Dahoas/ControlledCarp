import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/default/circle_prompt_finetuned.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)
#to modify:
#save folder
#review
#min vs max

#reviews = ["Grammar_Usage", "Incoherent", "Technical_Jargon", "Redundant", "Commonsense", "Encyclopedic"]
reviews = ['This is a story about a good person',
			'This is a story about a neutral person',
			'This is a story about an evil person',
]
experiments = [{'review': review, 'save_folder': f'ckpts/{review}_default_model', 'minimize': False} for review in reviews]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)