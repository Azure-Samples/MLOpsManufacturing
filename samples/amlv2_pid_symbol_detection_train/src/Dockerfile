FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

USER root

RUN adduser --disabled-password --gecos '' pythonmlops

WORKDIR /home/pythonmlops

COPY requirements.deploy.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install azure-cli

COPY . .

RUN flake8 app setup.py

RUN chown -R pythonmlops:pythonmlops ./
USER pythonmlops

WORKDIR /home/pythonmlops/app
