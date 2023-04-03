from config.config import DATA_DIR
import datasets
from pathlib import Path

def load_dataset(dataset_path, dataset_name):
    dataset = datasets.load_from_disk(Path.joinpath(DATA_DIR, dataset_name))
    print(dataset['train'][1])

if __name__ == '__main__':
    load_dataset('Bingsu/Cat_and_Dog', 'Cat_and_Dog')
