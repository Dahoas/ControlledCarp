import os
import re
import random

path = '/home/ubuntu/alex/kg_story_gen/data/rocstory_data'

roc_prompts = []

with open(os.path.join(path, 'valid.txt')) as f:
	lines = f.readlines()
	for story in lines:
		first_sentence = re.split('\.|\?|!', story)[0]
		roc_prompts.append(first_sentence)

with open(os.path.join(path, 'test.txt')) as f:
	lines = f.readlines()
	for story in lines:
		first_sentence = re.split('\.|\?|!', story)[0]
		roc_prompts.append(first_sentence)

with open(os.path.join(path, 'train.txt')) as f:
	lines = f.readlines()
	random.shuffle(lines)
	for story in lines:
		first_sentence = re.split('\.|\?|!', story)[0]
		roc_prompts.append(first_sentence)
		if len(roc_prompts) > 10000:
			break
random.shuffle(roc_prompts)
print(len(roc_prompts))
print(roc_prompts[0])
print(roc_prompts[1])
with open('roc_prompts.txt', 'w') as f:
	f.writelines([f'{story}\n' for story in roc_prompts])
