FROM python:3.9-slim
WORKDIR /pipeline
ADD requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY load_process.py /pipeline