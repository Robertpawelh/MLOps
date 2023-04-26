from argparse import ArgumentParser
from config.config import BASE_DIR, CONFIG_DIR
import yaml

def get_cfg_from_args(default_cfg_path, parent_parser=None, wandb=False):
    parser = ArgumentParser(parents=[parent_parser] if parent_parser is not None else [])
    parser.add_argument('--cfg', default=default_cfg_path, type=str, help='path to script config file')
    parser.add_argument('--global_cfg', default=f'{CONFIG_DIR}/global_params.yaml', type=str, help='path to global config file')
    parser.add_argument('--stage', default='default', type=str, help='dvc stage name')

    args = parser.parse_args()
    
    with open(f'{args.global_cfg}', 'r') as f:
        global_cfg = yaml.safe_load(f)

    with open(f'{args.cfg}', 'r') as f:
        full_cfg = yaml.safe_load(f)
        cfg = full_cfg['default']
        if args.stage != 'default':
            cfg.update(full_cfg[args.stage])

        cfg['global_cfg'] = global_cfg

    return cfg
