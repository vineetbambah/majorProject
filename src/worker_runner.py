from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model


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


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    print(f"[Rank {config['rank']}] entering {config['mode']} mode for {config['algo']}", flush=True)
    comm_ctx = None

    try:
        # Step 1: communication setup first.
        comm_ctx = algo_module.setup(config)
        print(
            f"[Rank {config['rank']}] setup done; comm_ctx keys: {list(comm_ctx.keys()) if isinstance(comm_ctx, dict) else type(comm_ctx)}",
            flush=True,
        )

        # Step 2: build model.
        try:
            model = model_module.build_model(config)
        except NotImplementedError as error:
            print(f"[Rank {config['rank']}] build_model placeholder hit: {error}", flush=True)
            model = {"model_tag": config["model"], "placeholder": True}

        # Step 3: run one tiny train step to get gradients.
        try:
            local_grad = model_module.train_step(model, config)
        except NotImplementedError as error:
            print(f"[Rank {config['rank']}] train_step placeholder hit: {error}", flush=True)
            local_grad = {"rank": config["rank"], "gradients": [0.0]}

        # Step 4: pass gradients to algorithm module.
        try:
            synced_grad = algo_module.average(local_grad, comm_ctx, config)
        except NotImplementedError as error:
            print(f"[Rank {config['rank']}] average placeholder hit: {error}", flush=True)
            synced_grad = local_grad

        print(f"[Rank {config['rank']}] tiny step complete; synced_grad={synced_grad}", flush=True)
    finally:
        # Step 5: teardown last.
        algo_module.teardown(comm_ctx)
        print(f"[Rank {config['rank']}] teardown done", flush=True)
