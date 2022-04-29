import csv
import re

def read_dataset_component(filepath):
	data = list()
	with open(filepath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			data.append(row[1])
	return data

passages = read_dataset_component('carp_data/train_stories.csv')

import random
random.shuffle(passages)
prompts = passages
count = 0
with open('circle_prompts.txt', 'w') as f:
	for prompt in prompts:
		if count >= 10000:
			break
		first_sentence = re.split('\.|\?|!', prompt)[0]
		if len(first_sentence.split(" ")) > 3 and len(first_sentence.split(" ")) < 6:
			if count == 9999:
				f.write(f'{first_sentence}')
			else:
				f.write(f'{first_sentence}\n')
			count+=1