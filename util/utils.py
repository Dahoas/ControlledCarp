import os

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

def get_model_path(model):
	cur_path = os.path.dirname(os.path.abspath(__file__))
	model_path = os.path.dirname(cur_path) + '/ckpts/' + model
	return model_path

def get_carp_config_path(filename):
	cur_path = os.path.dirname(os.path.abspath(__file__))
	cur_path = os.path.dirname(cur_path)
	model_path = os.path.dirname(cur_path) + '/magiCARP/configs/' + filename
	return model_path