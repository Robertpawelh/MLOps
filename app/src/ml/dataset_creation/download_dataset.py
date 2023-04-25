#!/usr/bin/env python3

from datasets import load_dataset, DatasetDict
from pathlib import Path
import config.config as cfg
import wandb

def download_from_huggingface(dataset_path, val_size=0.5):
    dataset = load_dataset(dataset_path)
    test_val = dataset['test'].train_test_split(test_size=val_size)

    train_test_val_dataset = DatasetDict({
        'train': dataset['train'],
        'test': test_val['train'],
        'val': test_val['test']}
    )

    train_test_val_dataset.save_to_disk(cfg.DATASET_PATH)

if __name__ == '__main__':
    run = wandb.init(project=cfg.PROJECT_NAME, 
               name=f'{cfg.DATASET_NAME}_classification',
               job_type='download_data',
               group=cfg.EXPERIMENT_GROUP_NAME)
    download_from_huggingface(cfg.HUGGINGFACE_DATASET_PATH)
    artifact = wandb.Artifact(cfg.DATASET_NAME, type='dataset')
    artifact.add_dir(cfg.DATASET_PATH)
    run.log_artifact(artifact)
