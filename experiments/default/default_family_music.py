import os
import yaml
from run import run

config_path = '/home/ubuntu/alex/pref_learning/ControlledCarp/configs/default/default_family_music.yml'
with open(config_path, 'r') as f:
	config = yaml.safe_load(f)
#to modify:
#save folder
#review
#min vs max

#reviews = ["Grammar_Usage", "Incoherent", "Technical_Jargon", "Redundant", "Commonsense", "Encyclopedic"]
reviews = ['This story is about family.',
		   'This story is about music.'
]
for review in reviews:
	config['review'] = review
	config['save_folder'] = f'ckpts/{review}_default_model'
	run(config)