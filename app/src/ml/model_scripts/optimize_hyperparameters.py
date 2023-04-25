import pprint
import config.config as paths_cfg
import yaml
import wandb
from src.ml.model_scripts.train import train

if __name__ == '__main__':
    with open(f'{paths_cfg.CONFIG_DIR}/params.yaml', 'r') as f: # TODO: make path an argument 
        sweep_config = yaml.safe_load(f)['optimize_hyperparameters']

    sweep_id = wandb.sweep(sweep_config['sweep_parameters'], project=paths_cfg.PROJECT_NAME)

    wandb.agent(sweep_id, train, count=sweep_config['count'])
