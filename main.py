from curses import meta
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
from yolo_model import YoloModel
import argparse
import time
from PIL import Image, ImageOps
import numpy as np
from utils import *
from output import MQTTClient, HTTPClient, Message
import os
import datetime
import gc
import yaml
import argparse
import socket
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="configuration.yaml",
    help="path to configuration file",
)
args = parser.parse_args()


HOSTNAME = socket.gethostname()
if "cam-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("cam-", "")

with open(args.config, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)


model_1_config = cfg.get("models").get("flower")
MODEL_1_WEIGHTS = model_1_config.get("weights_path")
MODEL_1_CLASSES = model_1_config.get("classes")
MODEL_1_IMG_SIZE = model_1_config.get("image_size")
MODEL_1_CONFIDENCE_THRESHOLD = model_1_config.get("confidence_threshold")
MODEL_1_IOU_THRESHOLD = model_1_config.get("iou_threshold")
MODEL_1_MARGIN = model_1_config.get("margin")


input_config = cfg.get("input")
INPUT_TYPE = input_config.get("type", "url")
INPUT_URL = None
INPUT_USERNAME = None
INPUT_PASSWORD = None

camera = None

if INPUT_TYPE == "url":
    input_config_server = input_config.get("server")
    if input_config_server is None:
        logging.error("No server configuration found")
        exit(1)
    INPUT_URL = input_config_server.get("url")
    if INPUT_URL is None:
        logging.error("No input url found")
        exit(1)
    INPUT_USERNAME = input_config_server.get("username")
    INPUT_PASSWORD = input_config_server.get("password")

elif INPUT_TYPE == "camera":
    from picamera2 import Picamera2

    camera = Picamera2()
    input_config_camera = input_config.get("camera")
    INPUT_CAMERA_WIDTH = input_config_camera.get("width", 4656)
    INPUT_CAMERA_HEIGHT = input_config_camera.get("height", 3496)
    cam_config = camera.still_configuration()
    cam_config["main"]["size"] = (INPUT_CAMERA_WIDTH, INPUT_CAMERA_HEIGHT)
    camera.configure(cam_config)
    camera.start()


# Output configuration
output_config = cfg.get("output")
IGNORE_EMPTY_RESULTS = output_config.get("ignore_empty_results", False)


# Output configuration (MQTT)
TRANSMIT_MQTT = False
mclient = None
if output_config.get("mqtt") is not None:
    output_config_mqtt = output_config.get("mqtt")
    if output_config_mqtt.get("transmit_mqtt", False):
        TRANSMIT_MQTT = True
        logging.info("Transmitting to MQTT")
        mqtt_host = output_config_mqtt.get("host")
        mqtt_port = output_config_mqtt.get("port")
        mqtt_topic = output_config_mqtt.get("topic")
        mqtt_topic = mqtt_topic.replace("${hostname}", HOSTNAME)
        mqtt_username = output_config_mqtt.get("username")
        mqtt_password = output_config_mqtt.get("password")
        mqtt_use_tls = output_config_mqtt.get("use_tls", mqtt_port == 8883)
        logging.info(
            "MQTT host: {}, port: {}, topic: {}, username {} use_tls: {}".format(
                mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_use_tls
            )
        )
        mclient = MQTTClient(
            mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_password, mqtt_use_tls
        )

# Output configuration (HTTP)
TRANSMIT_HTTP = False
hclient = None
if output_config.get("http") is not None:
    output_config_http = output_config.get("http")
    if output_config_http.get("transmit_http", False):
        TRANSMIT_HTTP = True
        logging.info("Transmitting to HTTP")
        http_url = output_config_http.get("url")
        http_url = http_url.replace("${hostname}", HOSTNAME)
        http_username = output_config_http.get("username")
        http_password = output_config_http.get("password")
        http_method = output_config_http.get("method", "POST")
        logging.info(
            "HTTP url: {}, method: {}, username: {}".format(
                http_url, http_method, http_username
            )
        )
        hclient = HTTPClient(http_url, http_username, http_password, http_method)

output_config = cfg.get("output")
OUTPUT_URL = output_config.get("url")
OUTPUT_USERNAME = output_config.get("username")
OUTPUT_PASSWORD = output_config.get("password")

CAPTURE_INTERVAL = cfg.get("capture_interval")


def capture_image():
    if INPUT_TYPE == "url":
        logging.info("Capturing image from {}".format(INPUT_URL))
        image = download_image(
            INPUT_URL, username=INPUT_USERNAME, password=INPUT_PASSWORD
        )
    elif INPUT_TYPE == "camera":
        logging.info("Capturing image from camera")
        np_array = camera.capture_array()
        image = Image.fromarray(np_array)
    return image


model_1 = YoloModel(
    MODEL_1_WEIGHTS,
    MODEL_1_IMG_SIZE,
    MODEL_1_CONFIDENCE_THRESHOLD,
    MODEL_1_IOU_THRESHOLD,
    classes=MODEL_1_CLASSES,
    margin=MODEL_1_MARGIN,
)



i = 0
while True:

    logging.info("downloading image {}".format(i))
    t0 = time.time()
    download_time = datetime.datetime.utcnow()
    image = capture_image()
    orig_width, orig_height = image.size
    t1 = time.time()
    logging.info("Getting image {} took {}".format(i, t1 - t0))
    model_1.reset_inference_times()
    msg = Message(download_time, HOSTNAME)
    crops, result_class_names, result_scores = model_1.get_crops(image)
    t2 = time.time()
    logging.info("processing step 1 image {} took {}".format(i, t2 - t1))
    print("result_class_names", result_class_names)
    nr_flowers = len(result_class_names)
    t2 = time.time()
    pollinator_index = 0
    for i in tqdm(range(nr_flowers)):
        crop_width, crop_height = crops[i].size
        msg.add_flower(
            i, result_class_names[i], result_scores[i], crop_width, crop_height
        )

            # print(result_class_names[i], result_class_names2[j], result_scores2[j])
    logging.info(
        "Found {} pollinators on {} flowers".format(pollinator_index, nr_flowers)
    )
    t3 = time.time()
    logging.info("processing step 2 image {} took {}".format(i, t3 - t2))
    logging.info(
        "Average processing time polli {}: {}".format(i, (t3 - t2) / max(len(crops), 1))
    )

    msg.add_metadata(model_1.get_metadata(multiple_inferences=False), [orig_width, orig_height])
    hclient.send_message(msg.construct_message())

    logging.info("TOTAL TIME: {}".format(t3 - t0))
    logging.info("Collecting")
    gc.collect()
    i += 1
    while time.time() - t0 < (CAPTURE_INTERVAL - 0.1):
        time.sleep(0.05)
