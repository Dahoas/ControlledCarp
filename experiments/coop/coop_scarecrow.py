import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/coop/scarecrow_coop.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)
#to modify:
#save folder
#review
#min vs max
experiments = [
	{
		'review': 'Grammar_Usage',
		'save_folder': 'ckpts/good_grammar_coop_model/',
		'minimize': True
	},
	{
		'review': 'Incoherent',
		'save_folder': 'ckpts/coherent_coop_model/',
		'minimize': True
	},
	{
		'review': 'Technical_Jargon',
		'save_folder': 'ckpts/technical_coop_model/',
		'minimize': False
	},
	{
		'review': 'Redundant',
		'save_folder': 'ckpts/redundant_coop_model/',
		'minimize': True
	},
	{
		'review': 'Commonsense',
		'save_folder': 'ckpts/commonsense_coop_model/',
		'minimize': True
	},
	{
		'review': 'Encyclopedic"',
		'save_folder': 'ckpts/not_encyclopedic_coop_model/',
		'minimize': True
	}
]
#reviews = ["Grammar_Usage", "Incoherent", "Technical_Jargon", "Redundant", "Commonsense", "Encyclopedic"]
for experiment in experiments:
	config['review'] = experiment['review']
	config['save_folder'] = experiment['save_folder']
	config['minimize'] = experiment['minimize']
	run(config)