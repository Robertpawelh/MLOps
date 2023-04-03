from config.config import DATA_DIR
from datasets import load_dataset, DatasetDict
from pathlib import Path

def download_dataset(dataset_path, dataset_name, val_size=0.5):
    dataset = load_dataset(dataset_path)
    test_val = dataset['test'].train_test_split(test_size=val_size)

    train_test_val_dataset = DatasetDict({
        'train': dataset['train'],
        'test': test_val['train'],
        'val': test_val['test']}
    )

    train_test_val_dataset.save_to_disk(Path.joinpath(DATA_DIR, dataset_name))

if __name__ == '__main__':
    download_dataset('Bingsu/Cat_and_Dog', 'Cat_and_Dog')
