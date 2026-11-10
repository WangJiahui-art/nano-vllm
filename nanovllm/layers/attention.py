import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# 根据给定的槽位映射, 把 Key 和 Value 写入到指定的缓存空间中.
@triton.jit  # 装饰器，表示下面的 Python 函数将由 Triton 编译器编译为在 GPU 上运行的代码
def store_kvcache_kernel(
    key_ptr,  # 指向当前需要被缓存的 Key 和 Value 数据的起始内存地址
    key_stride,  # 源数据中 Key 和 Value 数组的跨步（stride），用于处理数据在内存中的间隔。
    value_ptr,  # 指向当前需要被缓存的 Key 和 Value 数据的起始内存地址
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# 解析张量结构 数据对齐与合法性校验 启动底层GPU内核 "store_kvcache_kernel"
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # 初始化 KV Cache（键值缓存），初始为空张量，用于在推理时存储历史上下文的 K 和 V
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()  # 获取当前的全局上下文信息 #! 获取的是什么信息?
        k_cache, v_cache = self.k_cache, self.v_cache  # 获取当前对象中已有的 KV Cache

        if k_cache.numel() and v_cache.numel():
            # 如果 KV Cache 中已经有数据（numel() 不为0，说明不是第一次调用），则执行以下逻辑:
            # 将当前新传入的 k, v 根据上下文中的槽位映射（slot_mapping）存入（或更新）到 KV Cache 中
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 判断当前是否处于预填充（Prefill）阶段（即模型第一次处理输入的 Prompt 时）
        if context.is_prefill:

            # 如果上下文中包含分块表（block_tables），说明启用了前缀缓存（prefix cache）机制，直接复用缓存中的 KV
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache

            # 调用 FlashAttention 的变长序列计算函数，计算注意力结果
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode
            # 解码阶段 q 只有一个 token（长度为1），利用 cache 可以极大地加速推理速度

            # 为打印各个张量的维度和数据类型.
            # print(
            #     f"q: {q.unsqueeze(1), q.dtype}, k_cache : {k_cache.shape, k_cache.dtype}, \
            #       v_cache : {v_cache.shape, v_cache.dtype}, cache_seqlens : {context.context_lens.shape, context.context_lens.dtype}, \
            #       scale : {type(self.scale)}"
            # )
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
