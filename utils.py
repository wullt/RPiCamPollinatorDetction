# some parts of this code are taken from:
# https://github.com/karanjakhar/yolov5-export-to-raspberry-pi/blob/main/utils.py

from PIL import Image
from io import BytesIO
import requests
from requests.auth import HTTPBasicAuth
import base64
import datetime
import logging

RESIZE_RESAMPLE = Image.BOX


def download_image(url, username=None, password=None):
    auth = None
    if username is not None and password is not None:
        auth = HTTPBasicAuth(username, password)
    response = requests.get(url, stream=True, auth=auth)
    img = Image.open(BytesIO(response.content))
    return img


def upload_json(
    crops,
    classes,
    scores,
    url,
    username=None,
    password=None,
    record_date=datetime.datetime.utcnow(),
    metadata=None,
):
    auth = None
    if username is not None and password is not None:
        auth = HTTPBasicAuth(username, password)
    detected = []
    for i in range(len(crops)):
        bio = BytesIO()

        crops[i].save(bio, format="JPEG")
        img_str = base64.b64encode(bio.getvalue())
        detected.append(
            {
                "image": img_str.decode("utf-8"),
                "class": classes[i],
                "score": float(scores[i]),
            }
        )
    payload = {"detections": detected, "metadata": metadata}
    try:
        response = requests.post(url, json=payload, auth=auth)
        if response.status_code != 200:
            logging.error("upload failed: {}".format(response.text))
            return False
        return True

    except Exception as e:
        logging.error("upload failed: {}".format(e))
        return False


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), RESIZE_RESAMPLE)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
