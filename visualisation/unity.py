import socket
import json
import time
import math
import numpy as np
from model.vehicle.vehicle import Vehicle

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def sendPose(veh: Vehicle, sock: socket):

    state = veh.get_state()

    msg = {
        "t": time.time(),
        "pos": state[0:3].astype(float).tolist(),
        "quat": state[6:10].astype(float).tolist(),
    }

    sock.sendto(json.dumps(msg).encode("utf-8"), ('127.0.0.1', 49005))
