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
        # stage 1
        attention(num_tokens, hidden)
        # _, handles[0], a2e_event, a2e_isend_hook = buffer.a2e_isend_two_stage_v3(
        a2e_send_result[0] = buffer.a2e_isend_two_stage_v3(

            x,
            topk_idx,
            topk_weights,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )

        # stage 2
        attention(num_tokens, hidden)
        _, handles_0, a2e_event_0, a2e_isend_hook_0 = a2e_send_result[0]
        a2e_event_0.current_stream_wait()
        a2e_isend_hook_event_0 = a2e_isend_hook_0()
        a2e_isend_hook_event_0.current_stream_wait()
        a2e_send_result[1] = buffer.a2e_isend_two_stage_v3(
            x,
            topk_idx,
            topk_weights,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )

        # e2a_x, e2a_event, e2a_irecv_hook = buffer.e2a_irecv_two_stage_v3(
        e2a_recv_result[1] = buffer.e2a_irecv_two_stage_v3(
            topk_idx,
            topk_weights,
            handles_0,
            dispatch_use_fp8=use_fp8,
            out=None,
        )

        # stage 3
        attention(num_tokens, hidden)
        _, handles_1, a2e_event_1, a2e_isend_hook_1 = a2e_send_result[1]
        a2e_event_1.current_stream_wait()
        a2e_isend_hook_event_1 = a2e_isend_hook_1()
        a2e_isend_hook_event_1.current_stream_wait()
        a2e_send_result[2] = buffer.a2e_isend_two_stage_v3(
            x,
            topk_idx,
            topk_weights,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )

        e2a_x_1, e2a_event_1, e2a_irecv_hook_1  = e2a_recv_result[1]
        e2a_event_1.current_stream_wait()
        e2a_irecv_hook_event_1 = e2a_irecv_hook_1()
        e2a_irecv_hook_event_1.current_stream_wait()
        e2a_recv_result[2] = buffer.e2a_irecv_two_stage_v3(
            topk_idx,
            topk_weights,
            handles_1,
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        # loop
        for idx in range (0, 60):
            pre_mb = (idx + 2) %3
            mb = idx % 3
            # moe
            attention(num_tokens, hidden)

            # a2e wait
            _, handles, a2e_event, a2e_isend_hook = a2e_send_result[pre_mb]
            a2e_event.current_stream_wait()
            a2e_isend_hook_event = a2e_isend_hook()
            a2e_isend_hook_event.current_stream_wait()
            
            # a2e send
            a2e_send_result[mb] = buffer.a2e_isend_two_stage_v3(
                x,
                topk_idx,
                topk_weights,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )
            
            # e2a wait
            e2a_x, e2a_event, e2a_irecv_hook  = e2a_recv_result[pre_mb]
            e2a_event.current_stream_wait()
            e2a_irecv_hook_event = e2a_irecv_hook()
            e2a_irecv_hook_event.current_stream_wait()
            
            # e2a recv
            e2a_recv_result[mb] = buffer.e2a_irecv_two_stage_v3(
                topk_idx,
                topk_weights,
                handles,
                dispatch_use_fp8=use_fp8,
                out=None,
            )

    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:  
        dist.barrier()

        a2e_recv_result = [None] * 3
        e2a_send_result = [None] * 3
        # stage 1
        #     packed_recv_x_list[0],
        #     packed_recv_count,
        #     rdma_send_flags,
        #     handles[0],
        #     event,
        #     a2e_irecv_hook,
        a2e_recv_result[0] = buffer.a2e_irecv_two_stage_v3(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
        packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[0]
        event.current_stream_wait()
        a2e_irecv_hook_event = a2e_irecv_hook()
        a2e_irecv_hook_event.current_stream_wait()

        moe(num_tokens, hidden)
        # e2a_event, e2a_isend_hook = buffer.e2a_isend_two_stage_v3(
        e2a_send_result[0] = buffer.e2a_isend_two_stage_v3(
            packed_recv_x.clone(), 
            num_topk,
            handles,
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        a2e_recv_result[0] = buffer.a2e_irecv_two_stage_v3(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
        # stage 2
        moe(num_tokens, hidden)
        e2a_event, e2a_isend_hook = e2a_send_result[0]
        e2a_event.current_stream_wait()
        e2a_isend_event = e2a_isend_hook()
        e2a_isend_event.current_stream_wait()
        
        packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[0]
        e2a_send_result[1] = buffer.e2a_isend_two_stage_v3(
            packed_recv_x.clone(), 
            num_topk,
            handles,
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[0]
        event.current_stream_wait()
        a2e_irecv_hook_event = a2e_irecv_hook()
        a2e_irecv_hook_event.current_stream_wait()
        a2e_recv_result[1] = buffer.a2e_irecv_two_stage_v3(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )

        # stage 3
        moe(num_tokens, hidden)
        e2a_event, e2a_isend_hook = e2a_send_result[1]
        e2a_event.current_stream_wait()
        e2a_isend_event = e2a_isend_hook()
        e2a_isend_event.current_stream_wait()
        e2a_send_result[2] = buffer.e2a_isend_two_stage_v3(
            packed_recv_x.clone(), 
            num_topk,
            handles,
            dispatch_use_fp8=use_fp8,
            out=None,
        )
        packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[1]
        event.current_stream_wait()
        a2e_irecv_hook_event = a2e_irecv_hook()
        a2e_irecv_hook_event.current_stream_wait()        
        a2e_recv_result[2] = buffer.a2e_irecv_two_stage_v3(
            hidden,
            num_topk,
            num_max_tokens,
            num_experts,
            use_fp8=use_fp8,
        )
       
        # loop
        for idx in range (0, 60):
            pre_mb = (idx + 2) % 3
            mb = idx % 3
            # moe
            moe(num_tokens, hidden)
            # e2a wait
            e2a_event, e2a_isend_hook = e2a_send_result[pre_mb]
            e2a_event.current_stream_wait()
            e2a_isend_event = e2a_isend_hook()
            e2a_isend_event.current_stream_wait()

            # e2a send
            packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[pre_mb]
            e2a_send_result[mb] = buffer.e2a_isend_two_stage_v3(
                packed_recv_x.clone(), 
                num_topk,
                handles,
                dispatch_use_fp8=use_fp8,
                out=None,
            )

            # a2e wait
            packed_recv_x, packed_recv_count, rdma_send_flags, handles, event, a2e_irecv_hook = a2e_recv_result[pre_mb]
            event.current_stream_wait()
            a2e_irecv_hook_event = a2e_irecv_hook()
            a2e_irecv_hook_event.current_stream_wait()        

            # a2e recv
            a2e_recv_result[mb] = buffer.a2e_irecv_two_stage_v3(
                hidden,
                num_topk,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )
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