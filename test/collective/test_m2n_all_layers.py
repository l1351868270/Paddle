import contextlib
import random
from functools import partial
import time
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication import deep_ep

num_max_tokens = 512


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to("float32").view((x_fp8.shape[0], -1, 128))
    x_scales = x_scales.view((x_fp8.shape[0], -1, 1))
    return (x_fp32 * x_scales).view(x_fp8.shape).to("bfloat16")

A = paddle.randn((96, 8192), dtype="bfloat16")
B = paddle.randn((8192, 28672), dtype="bfloat16")
C = paddle.randn((28672, 8192), dtype="bfloat16")
def moe(num_tokens, hidden):
    paddle.matmul(paddle.matmul(A, B) + paddle.matmul(A, B), C)
    # time.sleep(1)
    return paddle.zeros((num_tokens, hidden), dtype="bfloat16")

def attention(num_tokens, hidden):
    paddle.matmul(paddle.matmul(A, B) + paddle.matmul(A, B), C)
    # time.sleep(1)
    return paddle.zeros((num_tokens, hidden), dtype="bfloat16")

def test_main(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    use_fp8: bool,
    rank: int,
    num_ranks: int,
    a_start_rank: int,
    a_num_ranks: int,
    e_start_rank: int,
    e_num_ranks: int,
    group: dist.communication.group,
    buffer: deep_ep.Buffer,
    seed: int = 0,
):
    paddle.seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % e_num_ranks == 0
    num_local_experts = num_experts // e_num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), 'Too many ranks (exceeding test precision limit)'

    GB = 192
    MB = 64
    num_micro_batches = GB // MB # 3
    num_hidden_layers = 2
    moe_layer_start_index = 0
    if rank >= a_start_rank and rank < a_start_rank + a_num_ranks:
        x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
            rank + 1
        )
        topk_idx = paddle.randint(
            0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
        )
        print(f"rank: {rank}, num_local_experts: {num_local_experts}")
        topk_weights = paddle.ones((num_tokens, num_topk), dtype="float32").abs_()
        print("x: ", x, flush=True)

        a2e_send_result = [None] * 3
        e2a_recv_result = [None] * 3

        dist.barrier()
 
        # loop
        for idx in range (moe_layer_start_index * num_micro_batches, num_hidden_layers * num_micro_batches):
            a2e_layer_idx_pre = (idx - 1) // 3
            a2e_mb_idx_pre = (idx - 1) % 3
            a2e_layer_idx = idx // 3
            a2e_mb_idx = idx % 3

            e2a_layer_idx = (idx - 2) // 3
            e2a_mb_idx = (idx - 2) % 3
            e2a_layer_idx_next = (idx - 1) // 3
            e2a_mb_idx_next = (idx - 1) % 3
            # attention
            print(f"====== compute attention {a2e_mb_idx}_{a2e_layer_idx}", flush=True)
            # attn 等待上一个micro batch数据接收完
            if a2e_layer_idx_pre >=  moe_layer_start_index:
                print(f"send attention {a2e_mb_idx_pre}_{a2e_layer_idx_pre} data end", flush=True)
            # attn 每一个micro batch均发送数据
            print(f"send attention {a2e_mb_idx}_{a2e_layer_idx} data begin", flush=True)
            # attn 最后一层不在接收数据
            if e2a_layer_idx >=  moe_layer_start_index and e2a_layer_idx < num_hidden_layers - 1:
                print(f"recv moe {e2a_mb_idx}_{e2a_layer_idx} data end", flush=True)
            # attn 最后一层不在接收数据
            if e2a_layer_idx_next >=  moe_layer_start_index and e2a_layer_idx_next < num_hidden_layers - 1:
                print(f"recv moe {e2a_mb_idx_next}_{e2a_layer_idx_next} data begin", flush=True)
        print(f"send attention {a2e_mb_idx}_{a2e_layer_idx} data end", flush=True)

    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:  
        dist.barrier()
        # loop
        # moe 第一次启动接收数据
        print(f"recv attention {0}_{moe_layer_start_index} data begin", flush=True)
        for idx in range (moe_layer_start_index * num_micro_batches, num_hidden_layers * num_micro_batches):
            a2e_layer_idx = idx // 3
            a2e_mb_idx = idx % 3
            a2e_layer_idx_next = (idx + 1) // 3
            a2e_mb_idx_next = (idx + 1) % 3

            e2a_layer_idx_pre = (idx - 1) // 3
            e2a_mb_idx_pre = (idx - 1) % 3
            e2a_layer_idx_pre_pre = (idx - 2) // 3
            e2a_mb_idx_pre_pre = (idx - 2) % 3
            # moe 最后一层不发送数据
            # moe 等待上上一个micro batch的数据
            if e2a_layer_idx_pre_pre >=  moe_layer_start_index and e2a_layer_idx_pre_pre < num_hidden_layers - 1:
                print(f"send moe {e2a_mb_idx_pre_pre}_{e2a_layer_idx_pre_pre} data end", flush=True)
            # moe 启动发送上一个micro batch的数据
            if e2a_layer_idx_pre >= moe_layer_start_index and e2a_layer_idx_pre < num_hidden_layers - 1:
                print(f"send moe {e2a_mb_idx_pre}_{e2a_layer_idx_pre} data begin", flush=True)
            # moe 每一个micro batch 都等待数据接收完
            print(f"recv attention {a2e_mb_idx}_{a2e_layer_idx} data end", flush=True)
            # moe 最后一个micro batch不再启动接收下一个数据
            if idx < num_hidden_layers * num_micro_batches - 1:
                print(f"recv attention {a2e_mb_idx_next}_{a2e_layer_idx_next} data begin", flush=True)
            print(f"====== compute moe {a2e_mb_idx}_{a2e_layer_idx}", flush=True)            

    dist.barrier()

def test_loop():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    a_start_rank = 0
    a_num_ranks = 16
    e_start_rank = a_start_rank + a_num_ranks
    e_num_ranks = num_ranks - a_num_ranks

    num_tokens, hidden, num_topk, num_experts = 96, 8192, 8, 64

    assert (
        num_tokens <= num_max_tokens
    ), "num_tokens must be less equal to num_max_tokens"
    num_rdma_ranks = num_ranks / 8
    num_local_experts = num_experts / num_ranks
    num_rdma_bytes = deep_ep.M2NBuffer.get_low_latency_rdma_size_hint_two_stage(
        num_max_tokens, hidden, num_ranks, a_num_ranks, e_num_ranks, num_experts, num_topk
    )
    
    use_fp8 = False
    num_nvl_bytes = deep_ep.M2NBuffer.get_low_latency_nvl_size_hint_two_stage(
        num_max_tokens, hidden, num_ranks, a_num_ranks, e_num_ranks, num_experts, num_topk, use_fp8
    )
    print(
        f'Allocating rdma buffer size: {num_rdma_bytes / 1e6} MB, nvl buffer size: {num_nvl_bytes / 1e6} MB...',
        flush=True,
    )

    buffer = deep_ep.M2NBuffer(
        group,
        a_start_rank,
        a_num_ranks,
        e_start_rank,
        e_num_ranks,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_rdma_ranks,
    )
    test_main(
        num_tokens,
        hidden,
        num_experts,
        num_topk,
        use_fp8,
        rank,
        num_ranks,
        a_start_rank,
        a_num_ranks,
        e_start_rank,
        e_num_ranks,
        group,
        buffer,
        seed=1,
    )


def init_dist_env(world_size, seed=20):
    context = contextlib.nullcontext()
    with context:
        # start to init distributed env
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

        fleet.init(is_collective=True, strategy=strategy)


if __name__ == '__main__':
    if dist.get_world_size() > 1:
        init_dist_env(dist.get_world_size())
    test_loop()