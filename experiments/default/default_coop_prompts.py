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

#reviews = ['characters laughing or finding things funny', 'imagery/descriptions', 'characters asking questions', 'praying/religion/church', 'accident/bad scenarios', 'character internal monologues and thoughts', 'crimes', 'fighting', 'music', 'family']

reviews = ['I liked how funny the characters are.',
		   'Good use of imagery.',
		   'They ask good questions.',
		   'I like the parts about religion.',
		   'This is about a tragic accident.',
		   'Nice internal monologue.',
		   'This is a story about crime.',
		   'This is a good fight scene',
		   'This is a story about music',
		   'This is a story about family',
]
experiments = [{'review': review, 'save_folder': f'ckpts/{review}_default_coop_prompts_model', 'minimize': False} for review in reviews]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)