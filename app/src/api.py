# app/api.py
from fastapi import FastAPI, Request
from http import HTTPStatus
from typing import Dict
from src.schemas import PredictionInput, PredictionOutput
from src.models.cnn import CNN
from pathlib import Path
from src.pipeline.predict import predict
from PIL import Image
from fastapi.exceptions import RequestValidationError
from src.exception_handler import validation_exception_handler, python_exception_handler
import io
import config.config as cfg
import requests
import uuid
import torch
import json

app = FastAPI(title='App', description='API for DL model.')
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(requests.exceptions.MissingSchema, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)

@app.on_event('startup')
async def _startup():
    print('Starting up...')
    model_parameters = json.load(open(cfg.MODEL_PARAMETERS_PATH))
    model = CNN(model_parameters)
    model.load_state_dict(torch.load(Path.joinpath(cfg.MODEL_DIR, 'model.pt'),
                                    map_location=cfg.device))
    model.eval()
    app.package = {
        'model': model
    }

# TODO: cache to avoid downloading the same file
@app.post('/predict',
         response_model=PredictionOutput)
async def _predict(request: Request, predictionInput: PredictionInput):
    response = requests.get(predictionInput.image_url)

    if not response.ok:
        print(response)

    with open(f'{cfg.IMAGES_DIR}/{uuid.uuid4()}.jpg', 'wb') as f:
        f.write(response.content)

    data = Image.open(io.BytesIO(response.content)).convert('RGB')
    y_pred = predict(app.package, data)

    response = {
        'status-code': HTTPStatus.OK,
        'cat': 1 - y_pred,
        'dog': y_pred,
    }

    return response
