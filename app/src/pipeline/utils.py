import torch
import numpy as np
import random

SEED = 1234

def set_seeds(seed=1234): # TODO: replace with seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
