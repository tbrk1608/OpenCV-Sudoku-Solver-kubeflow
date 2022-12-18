@echo off
SET IMAGE=opencv-sudokusolver-kubeflow
SET TAG=cnn_model
SET USERID=tbrk1608

docker build -t  %USERID%/%IMAGE%:%TAG% .
docker push %USERID%/%IMAGE%:%TAG%