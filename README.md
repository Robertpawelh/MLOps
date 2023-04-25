## MLops
This repository presents a simple project built on some of the tools from the MLOps stack - this is an example PyTorch model for image classification, for which an API was created. \
The project mainly uses PyTorch-Lightning, FastAPI, pytest, DVC, Docker, Github Actions, Weights&Biases.

This project is just an example of a combination of several tools. 
For more functionality, DVC should be connected to a remote repository.

## Running
If you don't have Docker installed, you can run the project locally. \
To do this, you need to install the required libraries from the requirements.txt file. \
Then you have to install the app package in editable mode.
```bash
pip3 install -r requirements.txt
pip3 install -e app --no-cache-dir
```
Remember to log to wandb:
```bash
wandb login
```
If you don't have models trained, you can train them using DVC pipelines:
```bash
dvc repro -R pipelines
```
Then you can run API:
```bash
uvicorn app.main:app --reload
```
