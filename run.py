import argparse
from util.utils import load_run_config
from carp_rl.finetune import finetune
from eval.evaluate import evaluate_model

def run(config):
    finetune(config)
    if config['EVAL']:
        evaluate_model(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/default.yml")
    args = parser.parse_args()
    config = load_run_config(args.config_path)
    run(config)