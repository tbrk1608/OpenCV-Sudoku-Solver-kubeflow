import argparse
import base64
import io
import os

import requests
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()
input_path = args.path
filename = os.path.basename(input_path)
with open(args.path, "rb") as image:
    encoded_image = base64.b64encode(image.read())

response_ = requests.post(
    "http://127.0.0.1:8080", json={"image": encoded_image.decode("utf-8")}
)
bytes_str = response_.json()["data"]
f = io.BytesIO(base64.b64decode(bytes_str))
solved_img = Image.open(f)
solved_img.save(f"solved_{filename}")
