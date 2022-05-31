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
experiments = [
	{
		'review': 'This story uses too much jargon',
		'save_folder': 'ckpts/technical_two_unfrozen_default_model/',
		'minimize': False
	},
	{
		'review': 'This story is redundant',
		'save_folder': 'ckpts/redundant_two_unfrozen_default_model/',
		'minimize': True
	},
	{
		'review': 'This story does not make sense',
		'save_folder': 'ckpts/commonsense_two_unfrozen_default_model/',
		'minimize': True
	},
	{
		'review': 'This story is too encyclopedic',
		'save_folder': 'ckpts/not_encyclopedic_two_unfrozen_default_model/',
		'minimize': True
	}
]
#reviews = ["Grammar_Usage", "Incoherent", "Technical_Jargon", "Redundant", "Commonsense", "Encyclopedic"]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)