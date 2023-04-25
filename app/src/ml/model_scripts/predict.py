import numpy as np
import torch
import config.config as cfg
import torchvision
from PIL import Image

def predict(package: dict, data: Image) -> np.ndarray:
    """
    Get prediction from model using input data
    :param package: dict from fastapi state with model
    :param data: list of input values
    :return: numpy array of model output
    """

    # process data
    #X = preprocess(package, input)
    model = package['model']
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), # TODO: refactor tranforms to be in on place and use the same parameters
                                    torchvision.transforms.ToTensor()])
    with torch.no_grad():
        # X = transform(data.convert("RGB"))
        X = transform(data)
        X = X.to(cfg.device)
        X = X.unsqueeze(0)

        y_pred = model(X)

    return y_pred.cpu().numpy()
