import argparse
import base64
import os
from pathlib import Path

import cv2
import imutils
import numpy as np
from flask import Flask, jsonify, request
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from norvig_algo import solve, squares

app = Flask(__name__)


def prepare_img(img):
    img = cv2.resize(img, (28, 28)).astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_sudoku_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            grid = four_point_transform(gray, approx.reshape(4, 2))
            return grid
        else:
            pass


def extract_cells(grid):
    cell_content = {}
    H, W = grid.shape  # H,W
    cell_height = H // 9
    cell_width = W // 9
    for rw in range(0, H, cell_height):
        if rw + cell_height < H:
            row = grid[rw : rw + cell_height]
            for cl in range(0, W, cell_width):
                if cl + cell_width < W:
                    cell = row[:, cl : cl + cell_width]
                    cell_th = cv2.threshold(
                        cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                    )[1]
                    cell_th = clear_border(cell_th)
                    digit = None
                    cnts = cv2.findContours(
                        cell_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cnts = imutils.grab_contours(cnts)
                    if not (len(cnts) == 0):
                        c = max(cnts, key=cv2.contourArea)
                        mask = np.zeros(cell_th.shape, dtype="uint8")
                        cv2.drawContours(mask, [c], -1, 255, -1)
                        h, w = cell_th.shape
                        percentFilled = cv2.countNonZero(mask) / float(w * h)
                        if percentFilled > 0.03:  # then not noise
                            digit = cv2.bitwise_and(cell_th, cell_th, mask=mask)
                    coords = (rw, rw + cell_height, cl, cl + cell_width)
                    cell_content[coords] = digit
    return cell_content


def get_str_grid(cells):
    pred_grid = []
    empty_cells = {}
    for sq, (coords, cell) in zip(squares, cells.items()):
        if cell is None:
            pred_grid.append(".")
            empty_cells[sq] = coords
        else:
            inp = prepare_img(cell)
            pred = MODEL.predict(inp).argmax(axis=1)[0] + 1
            pred_grid.append(str(pred))
    str_grid = "".join(pred_grid)
    return str_grid, empty_cells


@app.route("/", methods=["POST"])
def predict_():
    input_image = request.get_json()["image"]
    nparr = np.fromstring(base64.b64decode(input_image), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    grid = preprocess_sudoku_grid(image)
    cells = extract_cells(grid)
    str_grid, empty_cells = get_str_grid(cells)
    res = solve(str_grid, print_res=False)
    empty_cell_values = {coords: res[sq] for sq, coords in empty_cells.items()}

    # OUTPUT IMAGE
    copy_ = image.copy()
    for coords, value in empty_cell_values.items():
        org = (coords[2] + 20, coords[1] - 10)
        cv2.putText(
            img=copy_,
            text=value,
            org=org,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=1,
        )

    cv2.imwrite(SAVE_PATH, copy_)
    # solved_image = base64.b64encode(cv2.imencode(".jpg", copy_)[1].tobytes())
    # return jsonify(data=solved_image.decode("utf-8"))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str)
    # parser.add_argument("--save_path", type=str)

    # args = parser.parse_args()
    BASE_DIR = "gs://kubeflow-opencv-sudoku-solver/"
    # Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
    SAVE_PATH = os.path.join(BASE_DIR, "solved.png")
    MODEL = load_model(MODEL_PATH)
    app.run(debug=False, host="0.0.0.0", port=8080)
