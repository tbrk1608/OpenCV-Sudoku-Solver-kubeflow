import argparse
import json
import os
from pathlib import Path

from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

EPOCHS = 20
BATCH_SIZE = 128


def cnn_model_train(some_str):

    with open(os.path.join(DATA_PATH, "data.json")) as data_file:
        data = json.load(data_file)
    data = json.loads(data)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    optimizer = Adam(learning_rate=1e-1)

    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(
        Conv2D(32, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(
        Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(
        Conv2D(128, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(9, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    predictions = model.predict(x_test)

    print(
        classification_report(
            y_test.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[str(x) for x in range(1, 10)],
        )
    )

    model.save(MODEL_PATH, save_traces="h5")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str)
    # parser.add_argument("--model_path", type=str)

    # args = parser.parse_args()
    BASE_DIR = "gs://kubeflow-opencv-sudoku-solver/"
    # Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH = os.path.join(BASE_DIR, "data.json")
    MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
    cnn_model_train()
