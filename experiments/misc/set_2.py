import os
import yaml
from run import run

experiment_config_path = '/srv/share2/ahavrilla3/ControlledCarp/configs/set_2'
for filename in os.listdir(experiment_config_path):
	file = os.path.join(experiment_config_path, filename)
	print(f"Starting experiment {filename}")
	with open(file, 'r') as f:
		config = yaml.safe_load(f)
		run(config)