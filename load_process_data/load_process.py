import argparse
import json
import os
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist


def remove(digit, data, labels):
    idx = (labels != digit).nonzero()
    return data[idx], labels[idx]


def load_process_data(data_path):
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    x_train, y_train = remove(0, x_train, y_train)
    x_test, y_test = remove(0, x_test, y_test)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    data = {
        "x_train": x_train.tolist(),
        "y_train": y_train.tolist(),
        "x_test": x_test.tolist(),
        "y_test": y_test.tolist(),
    }

    data_json = json.dumps(data)

    with open(data_path, "w") as out_file:
        json.dump(data_json, out_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()
    Path(args.data_path).parent.mkdir(parents=True, exist_ok=True)
    load_process_data(args.data_path)
