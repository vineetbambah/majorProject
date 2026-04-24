import multiprocessing as mp

from worker_runner import run_worker


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
