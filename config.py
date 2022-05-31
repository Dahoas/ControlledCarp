from dataclasses import dataclass

@dataclass
class PrefConfig:
	#LM model args
	lm_name: str
	ref_lm_name: str
	tk_name: str
	num_layers_unfrozen: int
	save_model: bool
	save_folder: str
	use_lm_ckpt: bool
	lm_ckpt_path: str
	tokenizer_path: str

	#Carp model args
	carp_version: str
	carp_config_path: str
	carp_ckpt_path: str

	#Target review
	review: str

	#Data
	data_path: str

	#Training args
	steps: int = 20000
	batch_size: int = 32
	forward_batch_size: int = 16
	ppo_epochs: int = 4
	txt_in_len: int = 14
	txt_out_len: int = 60
	lr: float = 1.412e-5
	init_kl_coef: float = 0.2
	target: int = 25
	horizon: int = 10000
	gamma: float = 1.0
	lam: float = 0.95
	cliprange: float = 0.2
	cliprange_value: float = 0.2
	vf_coef: float = 0.15

	#Logging
	LOG: bool = True
	EVAL: bool = True
	num_eval_examples: int = 10

	#Min or max reward
	minimize: bool = False

	#Threshold reward
	do_thresh: bool = False
	thresh: float = 10.0

	@classmethod
	def from_dict(cls, config):
		return cls(**config)

	def to_dict(self):
		data = self.__dict__.copy()
		return data

