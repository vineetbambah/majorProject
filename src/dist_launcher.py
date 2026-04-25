import json
import socket
import time

from worker_runner import run_worker

CONFIG_PATH = "config.json"


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_left_right_neighbor_ip(left_peer_rank, right_peer_rank):
    config = load_json_config(CONFIG_PATH)
    ip_list = config["ip_list"]
    left_ip = ip_list[left_peer_rank]
    right_ip = ip_list[right_peer_rank]
    return left_ip, right_ip


def create_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def build_distributed_topology(algo, rank):
    if algo == "ring":
        config = load_json_config(CONFIG_PATH)
        local_ip = config["ip_list"][rank]
        base_port = int(config.get("base_port", 5000))
        left_ip = config["ip_list"][(rank - 1) % config["world_size"]]
        right_ip = config["ip_list"][(rank + 1) % config["world_size"]]
        listener = create_socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((local_ip, base_port))
        listener.listen(1)

        right_conn = create_socket()
        while True:
            try:
                right_conn.connect((right_ip, base_port))
                break
            except OSError:
                time.sleep(0.2)

        left_conn, _ = listener.accept()
        listener.close()
        return {
            "left_conn": left_conn,
            "right_conn": right_conn,
            "left_endpoint_info": {
                "peer_rank": left_peer_rank,
                "direction": "left",
                "transport": "socket",
            },
            "right_endpoint_info": {
                "peer_rank": right_peer_rank,
                "direction": "right",
                "transport": "socket",
            },
        }

    if algo == "tree":
        return {}

    if algo == "parameter_server":
        return {}

    raise ValueError("Unknown algo")

def launch_distributed(config):
    topo = build_distributed_topology(
        config["algo"],
        config["rank"],
    )

    worker_config = {
        **config,
        **topo,
    }
    run_worker(worker_config)