from util.utils import load_run_config
from util.carp_util import load_carp, scorer

config = load_run_config('configs/set_6/coherent_coop.yml')
carp = load_carp(config["carp_version"], config["carp_config_path"], config["carp_ckpt_path"])

story = 'I went to the store to get groceries. I bought some bread and an apple. Then I went home. At home I made dinner.'
review = 'This story has short sentences'

score = scorer([story], [review], mode=config['carp_version'])
print(score)