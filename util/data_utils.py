def load_prompts(data_path):
	with open(data_path, 'r') as f:
		prompts = f.readlines()
	return prompts