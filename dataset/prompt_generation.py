from black import token
import npu
from util.utils import Debugger
import re
from transformers import AutoTokenizer
import os

# not sure if i need this...
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from functools import reduce

# permutes the string so that we can get all variations for bad words
def permute_string(string, lower = False):
  spaces = [string, " " + string, string + " ", " " + string + " "]
  line_breaks = ["\n" + string, string + "\n", "\n " + string]
  if lower: return spaces + line_breaks
  return spaces + line_breaks + permute_string(string.lower(), True)


bad_words = list(map(permute_string, ["Â",'http','python','A:','(c)','[',']','xml','<','>','*','Copyright']))
for i in range(1,10):
	underscore_string = reduce(lambda a,b : a + b, ["_"] * i)
	bad_words += [" " + underscore_string, underscore_string]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bad_words_ids = list(map(lambda str: tokenizer(str).input_ids, bad_words))

def postprocessing(text):
	res = re.search(f'(http|python|(A:)|(\(c\))|\[|\]|\#|xml|\<|\>|[0-9]|Copyright)', text)
	if res is None:
		new_sample_text = re.sub('\s\s+', " ", text)
		new_sample_text = re.sub('\n', '', new_sample_text)
	else:
		new_sample_text = None
	return new_sample_text


def generate_prompts():
	model_id = '60ca2a1e54f6ecb69867c72c'
	npu.api("Mj4C8b56cl3Nb27Axp_7568HNvLwK_5408GZdoGsz48", deployed=True)

	#Inspired by https://thewritepractice.com/short-story-ideas/
	prompt_inputs = [['Once upon a time '],
					['A man walked into '],
					['It was a dark and '],
					['It once was said '],
					['The goose was quite happy, for '],
					['The magical castle sat upon the hill '],
					['There once was a tortoise and a hare '],
					['There once was a kitten who lost her mittens ' ],
					['A cat with a scar '],
					['A group of children come upon'],
					['My prodigal son recently '],
					['The ghost of my wife '],
					['She was deeply in love '],
					['The talented performer decided '],
					['The orphan inherited '],
					['They bumped into each other and '],
					['When they were nearly at their destination '],
					['It turns out the psycopath '],
					['Alex cried tears of '],
					['Louis interrogated him about '],
					['The former KGB spy Putin '],
					['They found her body '],
					['The aliens allied with '],
					['The lighthouse which stood for centuries'],
					['Mountains spread as far as the '],
					['Flying had always scared '],
					['Alfred could not believe' ],
					['Melissa felt she should have '],
					['The music made them feel '],
					]

	kwargs = {
			'response_length': 20, # how many response tokens to generate
			'remove_input': True, # whether to return your input
			'num_beams': 4,
			'repetition_penalty': 1.2,
			'no_repeat_ngram_size': 4,
			'early_stopping': False,
			'temperature': 1.3,
			'do_sample': True,
			'bad_word_ids' : bad_words_ids,
			'num_return_sequences': 25
			# all params from the transformers library `generate` function are supported
		}

	datafile = 'alt_prompts.txt'
	#Generation
	with open(datafile, 'r') as f:
		lines = f.readlines()
		n = len(lines)
	num_prompts = max(10000 - n, 0)
	num_batches = (num_prompts + len(prompt_inputs) - 1) // len(prompt_inputs)
	num_batches = (num_batches // kwargs['num_return_sequences']) + 1
	for _ in range(num_batches):
		for input in prompt_inputs:
			input_prompt = ["The following is a short fiction story: "]
			output_list = npu.predict(model_id, input_prompt, kwargs)[0]['generated_text']
			for output in output_list:
				processed_output = postprocessing(output)
				if processed_output is not None:
					with open(datafile, 'a+') as f:
						f.write(f'{processed_output}\n')

def load_prompts(data_file):
	cur_path = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.dirname(cur_path) + '/dataset/' + data_file
	with open(data_path, 'r') as f:
		prompts = f.readlines()
	return prompts

def clean_data(data_file):
	prompts = load_prompts(data_file)
	filtered_prompts = []
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	for i, prompt in enumerate(prompts):
		input_ids = tokenizer.encode_plus(prompt)['input_ids']
		if len(input_ids) >= 15:
			filtered_prompts.append(prompt)
	with open(data_file, 'w') as f:
		f.writelines(filtered_prompts)

def find_min_token_count(data_file):
	prompts = load_prompts(data_file)
	min_token_count = 20
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	for i, prompt in enumerate(prompts):
		n = len(tokenizer.encode_plus(prompt)['input_ids'])
		if n < min_token_count:
			min_token_count = n
	print(min_token_count)


if __name__ == "__main__":
	#generate_prompts()
	clean_data('alt_prompts.txt')
	#find_min_token_count('alt_prompts.txt')