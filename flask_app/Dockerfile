FROM python:3.9-slim
# cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /pipeline
ADD requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY norvig_algo.py /pipeline
COPY app.py /pipeline