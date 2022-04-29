# imports
import os
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from transformers import GPT2Tokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
from util.carp_util import scorer, load_carp
import re
from dataset.prompt_generation import generate_prompts
import matplotlib.pyplot as plt
import numpy as np
from carp.configs import CARPConfig

#Testing model generate
def test_1():
	#Model output
	#For some reason installing happytransformer seems to allow these imports to work
	model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
	tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
	query_txt = "This morning I went to the "
	#return_tensors=pt means pytorch
	#encode plus returns mask for sequence classification and overflowing elements
	#in addition to encode return
	query_tensor = tokenizer.encode_plus(query_txt, return_tensors="pt")
	print(query_tensor)
	##**dict unpacks dict into keyword arguments for function call
	#Not able to call generate if loading from AutoModel.
	#AutoModel vs AutoModelForCausalLM somehow diff
	output = model.generate(
			**query_tensor,
			num_beams = 3,
			repetition_penalty = 1.2,
			no_repeat_ngram_size = 4,
			early_stopping = True,
			num_return_sequences = 3
		)
	print(output)
	#output[0] for batch
	decoded_output = tokenizer.decode(output[0])
	print(decoded_output)

#Testing model output
def test_2():
	model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
	tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
	query_txt = "This morning I went to the "
	query_tensor = tokenizer.encode_plus(query_txt, return_tensors="pt")
	print(query_tensor)
	#output: CausalLMOutputWithPast is subclass of HF ModelOutput
	output = model(query_tensor['input_ids'])
	logits = output.logits #(batch_size, seq length, vocab_size)
	probs = F.softmax(logits[:, -1, :])
	print(probs.shape)
	#Multinomial sampling
	next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
	input_ids = torch.cat([query_tensor['input_ids'], next_token.unsqueeze(-1)], dim=-1)
	decoded_output = tokenizer.decode(input_ids[0])
	decoded_next_token = tokenizer.decode(next_token)
	print(decoded_output)
	print(decoded_next_token)

def test_get_model_path():
	model_path = get_model_path("CARP_L.pt")
	print(model_path)

def regex_test():
	sample_text = 'Once upon a time   \nwhen the stars were still in the sky.\n\nA'
	new_sample_text = re.sub('\s\s+', " ", sample_text)
	new_sample_text = re.sub('\n', '', new_sample_text)
	print(sample_text)
	print(new_sample_text)

def generation_test():
	generate_prompts()

def test_fine_tuned_lm():
	model_path = get_model_path('angry_sad_gpt2_model.pt')
	model = GPT2HeadWithValueModel.from_pretrained("lvwerra/gpt2-imdb")
	model.to('cuda')
	model.load_state_dict(torch.load(model_path))
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	query_txt = ["It was a wonderful day"]
	batch = query_txt
	batch_token =[tokenizer.encode(x, return_tensors="pt").to('cuda')[0, :15] for x in batch]
	query_tensors = torch.stack(batch_token)

	response_tensors = respond_to_batch(model, query_tensors, txt_len=40)
	stories = [tokenizer.decode(response_tensors[i, :]) for i in range(len(response_tensors))]
	print(stories)

def batch_encode_plus_test():
	prompt_inputs = ['There once was a boy ',
					 'The girl from Ipanema ']
	tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
	tokenized_prompts = tokenizer.batch_encode_plus(prompt_inputs)
	print(tokenized_prompts)


def test_plotting():
	y = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
	x = np.arange(len(y))
	plt.clf()
	plt.plot(x,y)
	plt.savefig('scores.png')

def magiCarp_test():
	carp_config_path = '/mnt/raid/users/AlexH/control_carp/magiCARP/configs'
	carp_config_file = 'carp_cloob.yml'
	carp_config_path = os.path.join(carp_config_path, carp_config_file)
	config = CARPConfig.load_yaml(carp_config_path)
	cloob_model = CARPCloob(config.model)
	model_path = get_model_path('CLOOB CARP Declutr B/')
	cloob_model.load(model_path)
	cloob_model = cloob_model.cuda()
	story = 'A man walked into a church and thanked God for all that was good in his life.'
	tokenized_story = cloob_model.passage_encoder.call_tokenizer(story).to('cuda')
	reviews = ['[quote] This story is too biblical.']
	tokenized_reviews = cloob_model.review_encoder.call_tokenizer(reviews).to('cuda')
	passage_batch = BatchElement(tokenized_story['input_ids'], tokenized_story['attention_mask'])
	review_batch = BatchElement(tokenized_reviews['input_ids'], tokenized_reviews['attention_mask'])

	with torch.no_grad():
		pass_encs, rev_encs = cloob_model.calculate_embeddings([passage_batch], [review_batch])
		confustion_matrix = cloob_model.cosine_sim(pass_encs[0], rev_encs[0])

	print(confustion_matrix)

def model_evaluation(model_name, model_ckpt):
	torch.cuda.empty_cache()
	model_path = get_model_path(model_ckpt)
	base_model = GPT2HeadWithValueModel.from_pretrained(model_name)
	base_model.to('cuda')
	tuned_model = GPT2HeadWithValueModel.from_pretrained(model_name)
	tuned_model.to('cuda')
	print(model_path)
	tuned_model.load_state_dict(torch.load(model_path))
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
	query_txt = [" A man and a woman are walking down the street. The man is",
				 "Today was my birthday and I spent it with my best friends doing ",
				 "The beaver had a happy family until a flood came and separated them. ",
				 'It was late afternoon and the sun was setting in the western sky. The ',
 				'A man and a woman are walking down the street. The man is',
 				'I first saw her when I was six years old. I dont remember much about it',
 				'I don’t usually read short fiction, I prefer to read long novels, but',
 				'A little while ago, I was working on a project in which I had to create a',
 				'Im trying to figure out how to make this work, but cant figure it out',
 				'A few weeks ago, I wrote a short story for my creative writing class. It was',
 				'The Story A young man sits at a desk, working on his homework. He',
 				'A few years ago, there was a boy who lived in a village. He was a']
	reviews = ['This is too suspenseful.', 'This is too biblical.']
	batch = query_txt
	batch_token =[tokenizer.encode(x, return_tensors="pt").to('cuda')[0, :14] for x in batch]
	query_txt = [tokenizer.decode(tokenized_text) for tokenized_text in batch_token]
	query_tensors = torch.stack(batch_token).to('cuda')

	tuned_response_tensors = respond_to_batch(tuned_model, query_tensors, txt_len=50)
	tuned_stories = [tokenizer.decode(tuned_response_tensors[i, :]) for i in range(len(tuned_response_tensors))]
	base_response_tensors = respond_to_batch(base_model, query_tensors, txt_len=50)
	base_stories = [tokenizer.decode(base_response_tensors[i, :]) for i in range(len(base_response_tensors))]

	carp_config_file = 'carp_l.yml'
	carp_config_path = get_carp_config_path(carp_config_file)
	carp_ckpt_path = get_model_path("CARP Roberta L/")
	carp = load_carp('default', carp_config_path, carp_ckpt_path)
	carp = carp.cuda()

	with open('results.txt', 'a') as f:
		f.write("Model Type: " + model_name + "\n")
		f.write("Reviews: " + str(reviews) + "\n")
		for prompt, (base, tuned) in zip(query_txt, zip(base_stories, tuned_stories)):
			base_story = prompt + " " + base
			tuned_story = prompt + " " + tuned
			base_score = scorer([base_story], reviews, carp)
			tuned_score = scorer([tuned_story], reviews, carp)
			f.write(f'Base model: {base_score}: ' + prompt + " " + base + "\n")
			f.write(f'Tuned model: {tuned_score}: ' + prompt + " " + tuned + "\n\n")


def model_statistics():
	model = GPT2HeadWithValueModel.from_pretrained("gpt2-large")
	print("MODEL PARAMS")
	#Iterate over raw parameters(weights and biases)
	#for p in model.parameters():
	#	print(p)
	#Iterating over modules i.e. groupings of parameters
	#last_modules = list(model.modules())[-10:]
	#for m in last_modules:
	#	print(m)
	#	for p in m.parameters():
	#		print(p)
	#Looking at GPT2 bLocks
	#for m in model.transformer.h:
	#	print(m)
	#
	#for m in model.transformer.ln_f:
	#	print(m)
	'''num_total_params = sum(p.numel() for p in model.parameters())
	num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Num total parameters: " + str(num_total_params))
	print("Num trainable parameters: " + str(num_trainable_params))'''

	'''model = ContrastiveModel(TextEncoder(), TextEncoder())
	num_total_params = sum(p.numel() for p in model.parameters())
	num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Num total parameters: " + str(num_total_params))
	print("Num trainable parameters: " + str(num_trainable_params))'''

def regex_test():
	res = re.search(f'(http|python|\(c\))', '(c)')
	print(res)

def test_carp_cloob():
	carp_config_path = '/mnt/raid/users/AlexH/control_carp/magiCARP/configs'
	carp_config_file = 'carp_cloob.yml'
	carp_config_path = os.path.join(carp_config_path, carp_config_file)
	config = CARPConfig.load_yaml(carp_config_path)
	cloob_model = CARPCloob(config.model)
	model_path = get_model_path('CLOOB CARP Declutr B/')
	cloob_model.load(model_path)
	cloob_model = cloob_model.cuda()

	story = 'The goose was quite happy, for it had just waddled into the pond. The duck was also happy, for it had just had a baby.'
	tokenized_story = cloob_model.passage_encoder.call_tokenizer(story).to('cuda')
	reviews = ['This story is too suspenseful.', 'This story is too scary.']

	score = scorer([story], reviews, cloob_model)
	print(score)

def pretrained_gpt_neo_test():
	model = GPTNeoHeadWithValueModel.from_pretrained("EleutherAI/gpt-neo-125M")
	model.to('cuda')
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	query_txt = [" A man and a woman are walking down the street. The man is",
				 "Today was my birthday and I spent it with my best friends doing ",
				 "The beaver had a happy family until a flood came and separated them. "]
	reviews = ["This story is too cheery."]
	batch = query_txt
	batch_token =[tokenizer.encode(x, return_tensors="pt").to('cuda')[0, :14] for x in batch]
	query_txt = [tokenizer.decode(tokenized_text) for tokenized_text in batch_token]
	query_tensors = torch.stack(batch_token).to('cuda')
	responses  = respond_to_batch(model, query_tensors, 15)
	tuned_stories = [tokenizer.decode(responses[i, :]) for i in range(len(responses))]
	for input, story in zip(query_txt, tuned_stories):
		print(input, story)

def load_yaml():
	import yaml
	with open('configs/default.yml', 'r') as f:
		config = yaml.safe_load(f)
	print(config)

def test_roc_gpt2():
	model = AutoModelForCausalLM.from_pretrained('ckpts/raw-roc-gpt2-large')
	num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(num_trainable_params)

	model.to('cuda')

	tokenizer = AutoTokenizer.from_pretrained('gpt2')
	text = "Tyrese was a man with very big lips. "
	model_input = tokenizer(text, return_tensors='pt').to('cuda')
	output = model.generate(**model_input, do_sample=True, max_length=100)
	print(output)
	decoded_output = tokenizer.decode(output[0])
	print(decoded_output)

def test_roc_prompts():
	with open('dataset/roc_prompts.txt','r') as f:
		lines = f.readlines()
	lengths = torch.tensor([len(line) for line in lines])
	print(torch.argmin(lengths))

def test_scarecrow_coop_model():
	carp_config_path = '/mnt/raid/users/AlexH/control_carp/magiCARP/configs'
	carp_config_file = 'carp_cloob.yml'
	carp_config_path = os.path.join(carp_config_path, carp_config_file)
	config = CARPConfig.load_yaml(carp_config_path)
	cloob_model = CARPCloob(config.model)
	model_path = get_model_path('CLOOB CARP Declutr B/')
	cloob_model.load(model_path)
	cloob_model = cloob_model.cuda()

if __name__=='__main__':
	print("STARTING TEST")
	#test_roc_gpt2()
	test_roc_prompts()