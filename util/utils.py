import os
import yaml

class Debugger:
	@classmethod
	def start(cls):
		with open('out.txt', 'w') as f:
			f.write("Started Debugging\n")

	@classmethod
	def write(cls, input):
		with open('out.txt', 'a+') as f:
			output = str(input)
			f.write(f'{output}\n')

def load_run_config(config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config