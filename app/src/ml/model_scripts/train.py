from src.ml.nn.cnn import CNN
from src.ml.model_utils.utils import set_seeds
from src.ml.dataset_loading.cat_dog_datamodule import CatAndDogDataModule
from typing import Dict
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import yaml
import config.config as paths_cfg
import multiprocessing
import pytorch_lightning as pl
import torch
import pytorch_lightning.loggers as pl_loggers
import wandb
from pprint import pprint


def train(model_parameters: Dict = None):
    log_model = 'all' if model_parameters is not None and model_parameters.get('log_model', False) else False
    
    wandb_logger = pl_loggers.WandbLogger(project=paths_cfg.PROJECT_NAME, 
                                          name=f'{paths_cfg.DATASET_NAME}_classification', 
                                          job_type='train',
                                          save_dir=paths_cfg.LOGS_DIR_WANDB,
                                          log_model=log_model,
                                          group=paths_cfg.EXPERIMENT_GROUP_NAME,
                                          config=model_parameters) # TODO: make it input to script

    model_parameters = wandb.config
    set_seeds(seed=model_parameters['seed'])
    num_workers = multiprocessing.cpu_count()
    model = CNN(**model_parameters)
    # wandb.watch(model, criterion, log="all", log_freq=10)  # track gradients

    trainer = pl.Trainer(accelerator='gpu' if paths_cfg.device.type == 'cuda' else 'cpu', 
                         devices=1,
                         max_epochs=model_parameters['epochs'],
                         callbacks=[EarlyStopping(monitor='val_loss', mode='min')],
                         logger=wandb_logger,
                         deterministic=True,
                         fast_dev_run=paths_cfg.FAST_DEV_RUN)
    
    # # dataset = wandb.use_artifact(f'{paths_cfg.DATASET_NAME}:latest')
    # # dataset = dataset.download(paths_cfg.DATASET_PATH)
    dm = CatAndDogDataModule(paths_cfg.DATASET_PATH, 
                             batch_size=model_parameters['batch_size'],
                             num_workers=num_workers)

    trainer.fit(model, dm)
    
    if model_parameters['log_model']:
        torch.save(model.state_dict(), paths_cfg.MODEL_PATH)
        art = wandb.Artifact(f'CNN-{wandb.run.id}.pt', type='model')
        art.add_file(paths_cfg.MODEL_PATH)
        wandb.log_artifact(art)

if __name__ == '__main__':
    with open(f'{paths_cfg.CONFIG_DIR}/params.yaml', 'r') as f: # TODO: make path an argument 
        config = yaml.safe_load(f)['train']
    train(config['model_parameters'])
