from src.models.cnn import CNN
from src.pipeline.utils import set_seeds
from src.model_utils.cat_dog_datamodule import CatAndDogDataModule

from pathlib import Path
from typing import Dict
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
import config.config as cfg
import multiprocessing
import pytorch_lightning as pl
import torch
import pytorch_lightning.loggers as pl_loggers

# TODO: connect parameters and logs with DVC
def train(model_parameters: Dict):
    set_seeds(seed=model_parameters['seed']) if 'seed' in model_parameters else set_seeds()
    num_workers = multiprocessing.cpu_count()
    model = CNN(model_parameters=model_parameters)

    logger = pl_loggers.CSVLogger(
                save_dir=cfg.LOGS_DIR,
                name='CNN')
    trainer = pl.Trainer(gpus=1,
                         max_epochs=model_parameters['epochs'],
                         callbacks=[EarlyStopping(monitor='val_loss', mode='min')],
                         logger=logger)
    dm = CatAndDogDataModule(cfg.DATASET_PATH,
                             batch_size=32,
                             num_workers=num_workers)

    trainer.fit(model, dm)
    torch.save(model.state_dict(), cfg.MODEL_PATH)

if __name__ == '__main__':
    model_parameters = json.load(open(cfg.MODEL_PARAMETERS_PATH))
    train(model_parameters)
