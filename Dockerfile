FROM ubuntu:22.04
WORKDIR /usr/src/app

ENV GIT_PYTHON_REFRESH=quiet

COPY app/setup.py app/setup.py
COPY app/src app/src
COPY requirements.txt requirements.txt
COPY dvc.lock dvc.lock
COPY dvc.yaml dvc.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.9 \
        python3-dev \
        python3-pip \
    && pip install --upgrade pip \
    && python3 -m pip install -e app --no-cache-dir \
    && pip install -r requirements.txt

COPY app/config app/config
COPY app/dvc_storage app/dvc_storage

RUN dvc init --no-scm
RUN dvc remote add -d storage app/dvc_storage
RUN dvc pull

EXPOSE 8000

ENTRYPOINT ["gunicorn", "-c", "app/src/api.py", "-k", "uvicorn.workers.UvicornWorker", "app.src.api:app"]
