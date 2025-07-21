import contextlib
import random
from functools import partial

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

def moe(num_tokens, hidden):
    return paddle.zeros((num_tokens, hidden), dtype="bfloat16")

def attention(num_tokens, hidden):
    return paddle.zeros((num_tokens, hidden), dtype="bfloat16")

def demo(
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
    MB = 64
    num_micro_batches = GB // MB
    num_hidden_layers = 54
    moe_layer_start_index = 3
    
    if rank > a_start_rank and rank < a_start_rank + a_num_ranks:
        x = paddle.ones((num_tokens, hidden), dtype="bfloat16") * (
            rank - rank_offset
        )
        x[:, -128:] = paddle.arange(0, num_tokens, dtype="bfloat16").view((-1, 1))
        topk_idx = paddle.randint(
            0, num_experts, shape=[num_tokens, num_topk], dtype="int64"
        )
        print(f"rank: {rank}, num_local_experts: {num_local_experts}")
        topk_weights = paddle.randn((num_tokens, num_topk), dtype="float32").abs_()
        print("x: ", x, flush=True)
        print("topk_idx: ", topk_idx, flush=True)
        print("topk_weights: ", topk_weights, flush=True)


        req_a2e_isend_cur = None
        req_e2a_irecv_cur = None
        for layer_idx in range(moe_layer_start_index, num_hidden_layers):
            # if use_fp8:
            #     num_scales = hidden // 128
            #     packed_recv_x_ = paddle.ones((num_local_experts,
            #                                   num_ranks * num_max_tokens,
            #                                   hidden),
            #                                   dtype='float8_e4m3fn')
                
            #     packed_recv_x_scales_ = paddle.ones((num_local_experts,
            #                                         num_scales,
            #                                         num_ranks * num_max_tokens),
            #                                         dtype='float32').transpose([0, 2, 1])
            #     packed_recv_x = (
            #         packed_recv_x_,
            #         packed_recv_x_scales_,
            #     )

            #     simulated_gemm_x = per_token_cast_back(
            #         packed_recv_x[0].view((-1, hidden)),
            #         packed_recv_x[1].contiguous().view((-1, hidden // 128)),
            #     ).view(packed_recv_x[0].shape)
            # else:
            #     packed_recv_x = paddle.ones((num_local_experts,
            #                                   num_ranks * num_max_tokens,
            #                                   hidden),
            #                                   dtype='bfloat16')
            #     simulated_gemm_x = packed_recv_x.clone()
        
            # handle = (
            #     None,
            #     None,
            #     None,
            #     None,
            #     None,
            #     None,
            #     num_experts,
            # )
            # e2a_x, event, req_e2a_irecv_cur = buffer.e2a_irecv_two_stage(
            #     simulated_gemm_x, 
            #     topk_idx,
            #     topk_weights,
            #     handle,
            #     dispatch_use_fp8=use_fp8,
            #     out=None,
            # )
                            
            for mb_idx in range(num_micro_batches):

                if layer_idx > moe_layer_start_index:
                    if req_e2a_irecv_cur is not None:
                        req_e2a_irecv_cur.wait()


                x = attention(num_tokens, hidden)
                # if layer_idx < num_hidden_layers:
                if req_a2e_isend_cur is not None:
                    req_a2e_isend_cur.wait()
                packed_recv_x, handle, event, req_a2e_isend_cur = buffer.a2e_isend_two_stage(
                    x,
                    topk_idx,
                    topk_weights,
                    num_max_tokens,
                    num_experts,
                    use_fp8=use_fp8,
                )
                
                if layer_idx == moe_layer_start_index:
                    if req_e2a_irecv_cur is None:
                        if use_fp8:
                            simulated_gemm_x = per_token_cast_back(
                                packed_recv_x[0].view((-1, hidden)),
                                packed_recv_x[1].contiguous().view((-1, hidden // 128)),
                            ).view(packed_recv_x[0].shape)
                        else:
                            simulated_gemm_x = packed_recv_x.clone()
                        
                        e2a_x, event, req_e2a_irecv_cur = buffer.e2a_irecv_two_stage(
                            simulated_gemm_x, 
                            topk_idx,
                            topk_weights,
                            handle,
                            dispatch_use_fp8=use_fp8,
                            out=None,
                        )

                if layer_idx > moe_layer_start_index:
                    if use_fp8:
                        simulated_gemm_x = per_token_cast_back(
                            packed_recv_x[0].view((-1, hidden)),
                            packed_recv_x[1].contiguous().view((-1, hidden // 128)),
                        ).view(packed_recv_x[0].shape)
                    else:
                        packed_recv_x = paddle.empty((num_local_experts,
                                                    num_ranks * num_max_tokens,
                                                    hidden),
                                                    dtype='bfloat16')
                        simulated_gemm_x = packed_recv_x.clone()
                
                    e2a_x, event, req_e2a_irecv_cur = buffer.e2a_irecv_two_stage(
                        simulated_gemm_x, 
                        topk_idx,
                        topk_weights,
                        handle,
                        dispatch_use_fp8=use_fp8,
                        out=None,
                    )



    if rank >= e_start_rank and rank < e_start_rank + e_num_ranks:
        req_e2a_isend_cur = None
        req_a2e_irecv_cur = None
        for layer_idx in range(moe_layer_start_index, num_hidden_layers):
            (
                packed_recv_x,
                packed_recv_count,
                rdma_send_flags,
                handle,
                event,
                req_a2e_irecv_cur,
            ) = buffer.a2e_irecv_two_stage(
                hidden,
                num_topk,
                num_max_tokens,
                num_experts,
                use_fp8=use_fp8,
            )

            for mb_idx in range(num_micro_batches):
                req_a2e_irecv_cur.wait()

                x = moe(num_tokens, hidden)

                if use_fp8:
                    simulated_gemm_x = per_token_cast_back(
                        packed_recv_x[0].view((-1, hidden)),
                        packed_recv_x[1].contiguous().view((-1, hidden // 128)),
                    ).view(packed_recv_x[0].shape)
                else:
                    simulated_gemm_x = packed_recv_x.clone()
                
                if req_e2a_isend_cur is not None:
                    req_e2a_isend_cur.wait()
                event, req_e2a_isend_cur = buffer.e2a_isend_two_stage(
                    simulated_gemm_x, 
                    num_topk,
                    handle,
                    dispatch_use_fp8=use_fp8,
                    out=None,
                )

                if layer_idx < num_hidden_layers - 1: 
                    (
                        packed_recv_x,
                        packed_recv_count,
                        rdma_send_flags,
                        handle,
                        event,
                        req_a2e_irecv_cur,
                    ) = buffer.a2e_irecv_two_stage(
                        hidden,
                        num_topk,
                        num_max_tokens,
                        num_experts,
                        use_fp8=use_fp8,
                    )
                
    paddle.device.synchronize()
    dist.barrier()
    run_time = 1
    print("run_time: ", run_time)
    print("num_experts: ", num_experts)

def test_demo():
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    group = paddle.distributed.new_group(range(num_ranks))
    print("rank: ", rank, flush=True)
    print("num_ranks: ", num_ranks, flush=True)

    a_start_rank = 0
    a_num_ranks = 16
    e_start_rank = a_start_rank + a_num_ranks
    e_num_ranks = num_ranks - a_num_ranks

    num_tokens, hidden, num_topk, num_experts = 96, 8192, 8, 24

    assert (
        num_tokens <= num_max_tokens
    ), "num_tokens must be less equal to num_max_tokens"
    num_rdma_ranks = num_ranks / 8
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
    demo(
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
    test_demo()