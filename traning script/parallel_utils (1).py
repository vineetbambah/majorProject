import torch

def ring_all_reduce(grad, left_conn, right_conn, world_size):
    total = grad.clone()
    send = grad.clone()

    for _ in range(world_size - 1):
        right_conn.send(send)
        recv = left_conn.recv()

        total += recv
        send = recv.clone()

    return total / world_size