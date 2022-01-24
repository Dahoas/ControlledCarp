import npu
from util.utils import Debugger
import re
import os

# not sure if i need this...
from transformers import GPT2Tokenizer
from functools import reduce

# permutes the string so that we can get all variations for bad words
def permute_string(string, lower = False):
  spaces = [string, " " + string, string + " ", " " + string + " "]
  line_breaks = ["\n" + string, string + "\n", "\n " + string]
  if lower: return spaces + line_breaks
  return spaces + line_breaks + permute_string(string.lower(), True)


bad_words = list(map(permute_string, ["Â"]))
for i in range(1,10):
	underscore_string = reduce(lambda a,b : a + b, ["_"] * i)
	bad_words += [" " + underscore_string, underscore_string]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bad_words_ids = list(map(lambda str: tokenizer(str).input_ids, bad_words))

def generate_prompts():
	model_id = '60ca2a1e54f6ecb69867c72c'
	npu.api("Mj4C8b56cl3Nb27Axp_7568HNvLwK_5408GZdoGsz48", deployed=True)

	prompt_inputs = [['Once upon a time '],
					['A man walked into '],
					['It was a dark and '],
					['It once was said '],
					['The goose was quite happy, for '],
					['The magical castle sat upon the hill '],
					['There once was a tortoise and a hare '],
					['There once was a kitten who lost her mittens ' ]
					]

	kwargs = {
			'response_length': 25, # how many response tokens to generate
			'remove_input': True, # whether to return your input
			'num_beams': 4,
			'repetition_penalty': 1.2,
			'no_repeat_ngram_size': 4,
			'early_stopping': True,
			'temperature': 1.3,
			'do_sample': True,
			'bad_word_ids' : bad_words_ids
			# all params from the transformers library `generate` function are supported
		}

	#Generation
	prompts = []
	num_prompts = 10000
	num_batches = (num_prompts + len(prompt_inputs) - 1) // len(prompt_inputs)
	for _ in range(num_batches):
		for input in prompt_inputs:
			print(input)
			input_prompt = ["The following is a short story: \n" + input[0]]
			output = npu.predict(model_id, input_prompt, kwargs)[0]
			text_output = input[0] + output['generated_text']
			new_sample_text = re.sub('\s\s+', " ", text_output)
			new_sample_text = re.sub('\n', '', new_sample_text)
			with open('prompts.txt', 'a+') as f:
				f.write(f'{new_sample_text}\n')
			prompts.append(new_sample_text)

	return prompts

def load_prompts():
	cur_path = os.path.dirname(os.path.abspath(__file__))
	data_file = 'prompts.txt'
	data_path = os.path.dirname(cur_path) + '/dataset/' + data_file
	with open(data_path, 'r') as f:
		prompts = f.readlines()
	return prompts

if __name__ == "__main__":
	generate_prompts()