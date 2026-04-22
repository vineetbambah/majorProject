import argparse
import json
import multiprocessing as mp

from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model

CONFIG_PATH = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tag-driven communication runner")
    parser.add_argument("--rank", type=int, required=True, help="Rank id")
    return parser.parse_args()


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_runtime_config(args: argparse.Namespace) -> dict:
    config = load_json_config(CONFIG_PATH)
    config["rank"] = args.rank
    return config


def validate_config(config: dict) -> None:
    required_keys = ["mode", "algo", "model", "lr", "world_size", "ip_list"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if config["mode"] not in {"local", "worker"}:
        raise ValueError("mode must be one of: local, worker")

    if config["algo"] not in {"ring", "tree", "parameter_server"}:
        raise ValueError("algo must be one of: ring, tree, parameter_server")

    if config["model"] not in {"ann", "cnn", "rnn"}:
        raise ValueError("model must be one of: ann, cnn, rnn")

    if config["world_size"] != len(config["ip_list"]):
        raise ValueError("world_size must match number of values in ip_list")

    if config["mode"] == "worker":
        if not 0 <= config["rank"] < config["world_size"]:
            raise ValueError("rank must be in range [0, world_size - 1]")


def get_algo_module(algo_tag: str):
    return {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }[algo_tag]


def get_model_module(model_tag: str):
    return {
        "ann": ann_model,
        "cnn": cnn_model,
        "rnn": rnn_model,
    }[model_tag]


def build_local_topology(algo, pipes, rank, world_size):
    if algo == "ring":
        return {
            "left_conn": pipes[rank][0],
            "right_conn": pipes[(rank + 1) % world_size][1],
        }

    if algo == "tree":
        return {}

    if algo == "parameter_server":
        return {}

    raise ValueError("Unknown algo")


def run_worker(config):
    algo_module = get_algo_module(config["algo"])

    print(f"[Rank {config['rank']}] entering {config['mode']} mode for {config['algo']}", flush=True)
    comm_ctx = algo_module.setup(config)
    print(f"[Rank {config['rank']}] setup done; comm_ctx keys: {list(comm_ctx.keys()) if isinstance(comm_ctx, dict) else type(comm_ctx)}", flush=True)

    algo_module.teardown(comm_ctx)
    print(f"[Rank {config['rank']}] teardown done", flush=True)


def launch_local(config):
    world_size = config["world_size"]
    algo = config["algo"]

    print(f"[parent rank 0] launching local {algo} setup for world_size={world_size}", flush=True)
    pipes = [mp.Pipe() for _ in range(world_size)]
    processes = []

    for rank in range(world_size):
        topo = build_local_topology(algo, pipes, rank, world_size)
        print(f"[parent rank 0] rank {rank} pipe endpoints assigned", flush=True)

        worker_config = {
            **config,
            "rank": rank,
            **topo,
        }

        process = mp.Process(target=run_worker, args=(worker_config,))
        process.start()
        processes.append(process)

    print("[parent rank 0] closing parent pipe endpoints", flush=True)
    for left_conn, right_conn in pipes:
        left_conn.close()
        right_conn.close()

    print("[parent rank 0] waiting for child processes", flush=True)
    for process in processes:
        process.join()
    print("[parent rank 0] local launch complete", flush=True)


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    validate_config(config)

    if config["mode"] == "local":
        if config["rank"] == 0:
            launch_local(config)
        return

    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    print("Final runtime config:")
    print(json.dumps(config, indent=2))
    print(f"Selected algorithm module: {algo_module.__name__}")
    print(f"Selected model module: {model_module.__name__}")
    print(f"Execution mode: {config['mode']}")


if __name__ == "__main__":
    main()