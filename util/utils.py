import os

def get_model_path(model):
	cur_path = os.path.dirname(os.path.abspath(__file__))
	model_path = os.path.dirname(cur_path) + '/ckpts/' + model
	return model_path