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
reviews = ['This was such a romantic story.',
		   'This story is magical.',
		   'This story is futuristic',
		   'This story is a horror',
		   'This story is dramatic',
		   'This story is cheesy',
		   'This story has good character development',
		   'This story has good dialogue',
		   'This story has a good setting',
		   'This is a funny story and made me laugh',
]
experiments = [{'use_lm_ckpt': False, 'review': review, 'save_folder': f'ckpts/{review}_new_nonfintuned_default_model', 'minimize': False} for review in reviews]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)