import pytest

from src.api import app
from fastapi.testclient import TestClient
from src.schemas import PredictionInput, PredictionOutput
from src.pipeline.predict import predict
import json

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_prediction_input():
    # TODO: move it to remote storage
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'
    predictionInput = PredictionInput(image_url=image_url)
    assert predictionInput.image_url == image_url

def test_prediction_output():
    predictionOutput = PredictionOutput(cat=0.3, dog=0.7)
    assert predictionOutput.cat == 0.3
    assert predictionOutput.dog == 0.7

def test_output_format(client):
    # TODO: move it to remote storage
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    output = response.json()
    assert 'cat' in output
    assert 'dog' in output

def test_output_probabilities_sum(client):
    # TODO: move it to remote storage
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    output = response.json()
    assert output['cat'] + output['dog'] == 1

def test_predict(client):
    # TODO: move it to remote storage
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    assert response.status_code == 200

def test_predict_with_png(client):
    # TODO: move it to remote storage
    image_url = 'https://w7.pngwing.com/pngs/174/600/png-transparent-cat-animal-lovely-cat.png'
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    assert response.status_code == 200

def test_predict_with_gif(client):
    # TODO: move it to remote storage
    image_url = 'https://media.tenor.com/fTTVgygGDh8AAAAM/kitty-cat-sandwich.gif'
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    assert response.status_code == 200

def test_predict_with_wrong_url(client):
    # TODO: move it to remote storage
    image_url = "blablabla.1332fa"
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    assert response.status_code == 422

def test_predict_with_wrong_url_type(client):
    image_url = 123
    predictionInput = PredictionInput(image_url=image_url)
    response = client.post("/predict", json=predictionInput.dict())
    assert response.status_code == 422

def test_predict_with_empty_input(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_without_image_url_input(client):
    response = client.post("/predict", json={'dog': 'cat'})
    assert response.status_code == 422

def test_predict_with_wrong_key_name(client):
    response = client.post("/predict", json={'img_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Dogge_Odin.jpg/800px-Dogge_Odin.jpg'})
    assert response.status_code == 422
