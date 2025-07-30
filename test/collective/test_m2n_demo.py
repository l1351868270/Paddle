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
    return paddle.zeros((num_tokens, hidden), dtype="bfloat16")

def attention(num_tokens, hidden):
    paddle.matmul(paddle.matmul(A, B) + paddle.matmul(A, B), C)
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
    num_rdma_ranks = num_ranks / 8

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert (
        num_ranks - rank_offset < 257
    ), 'Too many ranks (exceeding test precision limit)'

    GB = 192
    MB = 192
    # MB = GB
    num_micro_batches = GB // MB
    # num_hidden_layers = 54
    num_hidden_layers = 54
    moe_layer_start_index = 3
    # moe_layer_start_index = 1
    
    if rank >= a_start_rank and rank < a_start_rank + a_num_ranks:
        x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
            rank - rank_offset
        )
        x[:, -128:] = paddle.arange(0, num_tokens, dtype="bfloat16").view((-1, 1))
        x = paddle.randn((num_tokens, hidden), dtype="bfloat16")
        topk_idx = paddle.randint(
            0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
        )
        print(f"rank: {rank}, num_local_experts: {num_local_experts}")
        topk_weights = paddle.randn((num_tokens, num_topk), dtype="float32").abs_()
        print("x: ", x, flush=True)
        print("topk_idx: ", topk_idx, flush=True)
        print("topk_weights: ", topk_weights, flush=True)

        a2e_isend_req_cur = None
        e2a_irecv_events = [None] * num_micro_batches
        e2a_irecv_reqs = [None] * num_micro_batches
        e2a_xs = [None] * num_micro_batches

        a2e_irecv_reqs = [None] * num_micro_batches
        e2a_irecv_req_cur = None
        a2e_handles = [None] * num_micro_batches
        a2e_packed_recv_xs = [None] * num_micro_batches
        a2e_packed_recv_counts = [None] * num_micro_batches
        a2e_rdma_send_flagss = [None] * num_micro_batches

        a2e_isend_req = None
        handles = [None] * 3
        # 1_1
        attention(num_tokens, hidden)
        time.sleep(0.1)
        print(f"rank: {rank}, attention 1_1", flush=True)
        packed_recv_x, handles[0], event, a2e_isend_req = buffer.a2e_isend_two_stage_v2(
            x,
            topk_idx,
            topk_weights,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
        
        print(f"rank: {rank}, dispatch_send", flush=True)
        attention(num_tokens, hidden)
        time.sleep(0.1)
        print(f"rank: {rank}, attention 2_1", flush=True)
        event.current_stream_wait()
        print(f"rank: {rank}, dispatch_send event wait", flush=True)

        e2a_x, e2a_event, e2a_irecv_req = buffer.e2a_irecv_two_stage_v2(
            topk_idx,
            topk_weights,
            handles[0],
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        print(f"rank: {rank}, combine", flush=True)
        e2a_event.current_stream_wait()
        print(f"rank: {rank}, combine event wait", flush=True)


        # packed_recv_x, handles[1], event, a2e_isend_req = buffer.a2e_isend_two_stage_v2(
        #     x,
        #     topk_idx,
        #     topk_weights,
        #     num_max_tokens,
        #     num_experts,
        #     use_fp8=use_fp8,
        # )
        # print(f"rank: {rank}, dispatch_send", flush=True)
        
        # attention(num_tokens, hidden)
        # time.sleep(0.1)
        # print(f"rank: {rank}, attention 3_1", flush=True)

        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch_send event wait", flush=True)
        # packed_recv_x, handles[2], event, a2e_isend_req = buffer.a2e_isend_two_stage_v2(
        #     x,
        #     topk_idx,
        #     topk_weights,
        #     num_max_tokens,
        #     num_experts,
        #     use_fp8=use_fp8,
        # )
        # print(f"rank: {rank}, dispatch_send", flush=True)

        # # e2a_x, e2a_event, e2a_irecv_req = buffer.e2a_irecv_two_stage_v2(
        # #     topk_idx,
        # #     topk_weights,
        # #     handles[0],
        # #     dispatch_use_fp8=use_fp8,
        # #     out=None,
        # # )
        # # print(f"rank: {rank}, combine", flush=True)

        # attention(num_tokens, hidden)
        # time.sleep(0.1)
        # print(f"rank: {rank}, attention 1_2", flush=True)
        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch_send event wait", flush=True)
        # packed_recv_x, handles[2], event, a2e_isend_req = buffer.a2e_isend_two_stage_v2(
        #     x,
        #     topk_idx,
        #     topk_weights,
        #     num_max_tokens,
        #     num_experts,
        #     use_fp8=use_fp8,
        # )
        # print(f"rank: {rank}, dispatch_send", flush=True)
        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch_send event wait", flush=True)

        run_time = 1
        print("run_time: ", run_time)
        print("num_experts: ", num_experts)

    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:
        e2a_isend_req = None
        a2e_irecv_reqs = [None] * num_micro_batches
        a2e_handles = [None] * num_micro_batches
        a2e_packed_recv_xs = [None] * num_micro_batches
        a2e_packed_recv_counts = [None] * num_micro_batches
        a2e_rdma_send_flagss = [None] * num_micro_batches
        a2e_irecv_req = None     
        handles = [None] * 3
        # (
        #     packed_recv_x,
        #     packed_recv_count,
        #     rdma_send_flags,
        #     handles[0],
        #     event,
        #     a2e_irecv_req,
        # ) = buffer.a2e_irecv_two_stage_v2(
        #     hidden,
        #     num_topk,
        #     num_max_tokens,
        #     num_experts,
        #     use_fp8=use_fp8,
        # )
        # print(f"rank: {rank}, dispatch", flush=True)
        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch event wait", flush=True)
        # event = a2e_irecv_req.hook()
        # print(f"rank: {rank}, dispatch_recv", flush=True)
        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch_recv event wait", flush=True)
        # # time.sleep(0.1)
        (
            packed_recv_x,
            packed_recv_count,
            rdma_send_flags,
            handles[1],
            event,
            a2e_irecv_req,
        ) = buffer.a2e_irecv_two_stage_v2(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
        print(f"rank: {rank}, dispatch", flush=True)

        event.current_stream_wait()
        print(f"rank: {rank}, dispatch event wait", flush=True)

        moe(num_tokens, hidden)
        time.sleep(0.1)
        print(f"rank: {rank}, moe 1_1", flush=True)

        if use_fp8:
            simulated_gemm_x = per_token_cast_back(
                packed_recv_x[0].view((-1, hidden)),
                packed_recv_x[1].contiguous().view((-1, hidden // 128)),
            ).view(packed_recv_x[0].shape)
        else:
            simulated_gemm_x = packed_recv_x.clone()
        e2a_event, req = buffer.e2a_isend_two_stage_v2(
            simulated_gemm_x, 
            num_topk,
            handles[1],
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        print(f"rank: {rank}, combine", flush=True)
        e2a_event.current_stream_wait()
        print(f"rank: {rank}, combine wait", flush=True)
        # (
        #     packed_recv_x,
        #     packed_recv_count,
        #     rdma_send_flags,
        #     handle,
        #     event,
        #     a2e_irecv_req,
        # ) = buffer.a2e_irecv_two_stage_v2(
        #     hidden,
        #     num_topk,
        #     num_max_tokens,
        #     num_experts,
        #     use_fp8=use_fp8,
        # )
        # print(f"rank: {rank}, dispatch", flush=True)

        # moe(num_tokens, hidden)
        # time.sleep(0.1)
        # print(f"rank: {rank}, moe 2_1", flush=True)

        # event.current_stream_wait()
        # print(f"rank: {rank}, dispatch event wait", flush=True)
        # # (
        # #     packed_recv_x,
        # #     packed_recv_count,
        # #     rdma_send_flags,
        # #     handle,
        # #     event,
        # #     a2e_irecv_req,
        # # ) = buffer.a2e_irecv_two_stage_v2(
        # #     hidden,
        # #     num_topk,
        # #     num_max_tokens,
        # #     num_experts,
        # #     use_fp8=use_fp8,
        # # )
        # # print(f"rank: {rank}, dispatch", flush=True)

        # moe(num_tokens, hidden)
        # time.sleep(0.1)
        # print(f"rank: {rank}, moe 3_1", flush=True)

        # # event.current_stream_wait()
        # # print(f"rank: {rank}, dispatch event wait", flush=True)

        # # moe(num_tokens, hidden)
        # # time.sleep(0.1)
        # # print(f"rank: {rank}, moe 1_2", flush=True)
        
        # # e2a_x, e2a_event, e2a_irecv_req = buffer.e2a_irecv_two_stage_v2(
        # #     topk_idx,
        # #     topk_weights,
        # #     handles[0],
        # #     dispatch_use_fp8=use_fp8,
        # #     out=None,
        # # )
        # # print(f"rank: {rank}, combine", flush=True)

        # # e2a_event.current_stream_wait()
        # # print(f"rank: {rank}, combine wait", flush=True)
    run_time = 1
    print("run_time: ", run_time)
    print("num_experts: ", num_experts)
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