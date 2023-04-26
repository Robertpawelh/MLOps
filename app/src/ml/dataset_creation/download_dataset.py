#!/usr/bin/env python3

from datasets import load_dataset, DatasetDict
from pathlib import Path
from src.utils.config_parser import get_cfg_from_args
from config.config import BASE_DIR, WANDB_DIR
import yaml
import wandb

def download_from_huggingface(hf_dataset_path, dset_output_path, val_size=0.5):
    dataset = load_dataset(hf_dataset_path)
    test_val = dataset['test'].train_test_split(test_size=val_size)

    train_test_val_dataset = DatasetDict({
        'train': dataset['train'],
        'test': test_val['train'],
        'val': test_val['test']}
    )
    
    Path.mkdir(dset_output_path, parents=True, exist_ok=True)
    train_test_val_dataset.save_to_disk(dset_output_path)

if __name__ == '__main__':
    cfg = get_cfg_from_args(default_cfg_path=f'{BASE_DIR}/pipelines/download_data/params.yaml',)
    dset_output_path = Path(f'{BASE_DIR}/{cfg["data_dir"]}/{cfg["dataset_name"]}')
    run = wandb.init(project=cfg['global_cfg']['wandb']['project_name'], 
               name=f'{cfg["dataset_name"]}_classification',
               group=cfg['experiment_group_name'],
               job_type='download_data',
               dir=WANDB_DIR
               )
    
    download_from_huggingface(cfg['hf_dataset_path'], dset_output_path)

    # artifact = wandb.Artifact(cfg['dataset_name'], type='dataset')
    # artifact.add_dir(dset_output_path)
    # run.log_artifact(artifact)
