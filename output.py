import json
import datetime
from PIL import Image
from io import BytesIO
import base64
from dataclasses import dataclass
import os
import sys

import logging
import ssl
import requests

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
log.addHandler(handler)


class Message:
    def __init__(self, ts, node_id):
        self.flowers = []
        self.pollinators = []
        self.timestamp = ts
        self.metadata = {}
        self.node_id = node_id

    def add_flower(self, index, class_name, score, width, height):
        self.flowers.append(
            {
                "index": index,
                "class_name": class_name,
                "score": float(score),
                "width": width,
                "height": height,
            }
        )

    def add_pollinator(
        self, index, flower_index, class_name, score, width, height, crop=None
    ):
        pollinator = {
            "index": index,
            "flower_index": flower_index,
            "class_name": class_name,
            "score": float(score),
            "width": width,
            "height": height,
        }
        if crop is not None:
            bio = BytesIO()
            crop.save(bio, format="JPEG")
            pollinator["crop"] = base64.b64encode(bio.getvalue()).decode("utf-8")
        self.pollinators.append(pollinator)

    def add_metadata(self, flowermeta, pollimeta, input_image_size):
        self.metadata["flower_inference"] = flowermeta
        self.metadata["flower_inference"]["capture_size"] = input_image_size
        self.metadata["pollinator_inference"] = pollimeta

    def construct_message(self):
        message = {
            "flowers": self.flowers,
            "pollinators": self.pollinators,
            "timestamp": str(self.timestamp),
            "node_id": self.node_id,
            "metadata": self.metadata,
        }
        return message


class MQTTClient:
    def __init__(self, host, port, topic, username, password, use_tls):
        self.host = host
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.use_tls = use_tls
        if self.username is not None and self.password is not None:
            self.auth = {
                "username": self.username,
                "password": self.password,
            }
        else:
            self.auth = None

    def publish(self, message, filename=None, node_id=None, hostname=None):
        import paho.mqtt.publish as publish

        topic = self.topic
        if filename is not None:
            topic = topic.replace("${filename}", filename)
        if node_id is not None:
            topic = topic.replace("${node_id}", node_id)
        if hostname is not None:
            topic = topic.replace("${hostname}", hostname)
        log.info("Publishing to {} on topic: {}".format(self.host, topic))
        tls_config = None
        if self.use_tls:
            tls_config = {
                "certfile": None,
                "keyfile": None,
                "cert_reqs": ssl.CERT_REQUIRED,
                "tls_version": ssl.PROTOCOL_TLSv1_2,
                "ciphers": None,
            }

        publish.single(
            topic,
            json.dumps(message),
            1,
            auth=self.auth,
            hostname=self.host,
            port=self.port,
            tls=tls_config,
        )


class HTTPClient:
    def __init__(self, url, username, password, method="POST"):
        self.url = url
        self.username = username
        self.password = password
        self.method = method
        if self.username is not None and self.password is not None:
            self.auth = (self.username, self.password)
        else:
            self.auth = None

    def send_message(self, message, filename=None, node_id=None, hostname=None):
        headers = {"Content-type": "application/json"}
        url = self.url
        if filename is not None:
            url = url.replace("${filename}", filename)
        if node_id is not None:
            url = url.replace("${node_id}", node_id)
        if hostname is not None:
            url = url.replace("${hostname}", hostname)
        log.info("Sending results to {}".format(url))

        if self.auth is not None:
            headers["Authorization"] = "Basic " + base64.b64encode(
                bytes(self.auth[0] + ":" + self.auth[1], "utf-8")
            ).decode("utf-8")
        try:
            response = requests.request(
                self.method, url, headers=headers, data=json.dumps(message) , timeout=10
            )
            if response.status_code == 200:
                log.info("Successfully sent results to {}".format(url))
                return True
            else:
                log.error(
                    "Failed to send results to {}, status code is {}".format(
                        url, response.status_code
                    )
                )
                return False
        except Exception as e:
            log.error(e)
            return False
