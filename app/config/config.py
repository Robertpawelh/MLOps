from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
DATA_DIR = Path(BASE_DIR, 'data')
MODEL_DIR = Path(BASE_DIR, 'model')
IMAGES_DIR = Path(BASE_DIR, 'images')
LOGS_DIR = Path(BASE_DIR, 'logs')

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = 'Cat_and_Dog'
MODEL_NAME = 'model.pt'

DATASET_PATH = Path.joinpath(DATA_DIR, DATASET_NAME)
MODEL_PATH = Path.joinpath(MODEL_DIR, MODEL_NAME)
MODEL_PARAMETERS_PATH = Path.joinpath(CONFIG_DIR, 'model_parameters.json')

DVC_STORAGE = Path(BASE_DIR, 'dvc_storage')
DVC_STORAGE.mkdir(parents=True, exist_ok=True)

cuda = False
device = torch.device('cuda' if (torch.cuda.is_available() and cuda) else 'cpu')
torch.set_default_tensor_type("torch.FloatTensor")
if device.type == 'cuda':
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# NOTE: path to dataset?