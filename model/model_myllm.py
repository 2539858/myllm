# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MyLLM Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "MyLLM"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn



# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MyLLM Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 防止分母为0
        self.eps = eps
        # 可学习参数 γ， 归一化后，给每个维度一个可调的缩放
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # .pow(2)对每个元素求平方   .mean(-1, keepdim=True) 在最后一个维度求均值  rsqrt相当于1 / torch.sqrt(a)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # float()让数值更稳定，防止RMS精度不够  再转回原本x的类型
        return self.weight * self._norm(x.float()).type_as(x)

# 生成所有位置编码所需的频率和三角函数矩阵
# dim：每个attention head的维度；end：最大位置长度；rope_base：控制不同维度旋转频率的衰减速度, rope_scaling：长上下文扩展策略
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    # torch,arrange(0, dim, 2) 取[0, 2, ···, dim-2], :dim//2表示我们只要前 dim//2个元素，当dim为奇数的时候有用
    # .float 转换成float32，方便进行除法操作 最终得到 1/ base^(2i/d)
    freqs  = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0
    # rope_scaling 为超长的上下文准备
    if rope_scaling is not None:
        # 分别拿到，最大上下文长度、缩放因子、快Beta参数、慢Beta参数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        # 如果超出大小。进行复杂缩放 使用YaRN 的“频率平滑缩放”  现在用的最大长度 > 模型原始训练长度，只压低“高频维度”的旋转速度，保留低频维度的长程建模能力
        if end / orig_max > 1.0:
            # 反向映射函数/ 两个方程得到i的值 b是输入的值
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # [0, low)完全不缩放 (low, high) 从0-1线性缩放， (high, end)完全缩放
            # 确定从哪一维度开始进行平滑缩放，输入阈值beta_fast，计算维度索引i，floor取不超过该值的最小整数，max(···，0)保证下界不小于0
            low = max(math.floor(inv_dim(beta_fast)), 0)
            # 确定平滑缩放结束的维度索引，输入阈值beta_low,计算维度索引i，ceil向上取整，min(···,dim//2 - 1)保证上界不超过可用维度范围
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 构造线性斜坡，算出 γ 因子对于每一个 RoPE 维度索引 i 的值。
            # torch.arange(dim // 2, device=freqs.device).float()生成[0,2···,dim//2-1]，
            # -low让平滑过渡区起点为0 除以high-low，得到0-1的比例γ，max是为了防止除以0
            # torch.clamp(···,0,1)把γ限制在(0,1)，i < low → γ = 0,i > high → γ = 1,low ≤ i ≤ high → γ 线性增加
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            freqs = freqs * (1 - ramp + ramp / factor)
    # 生成[0,1,2···,end-1]序列表示token的索引t
    t = torch.arange(end, device=freqs.device)
    # 第i行第j列进行，a_i * b_j 的外积操作
    freqs = torch.outer(t, freqs).float()
    # RoPE 是“二维旋转”，每两个 hidden 维度共享同一个频率。 (x_2i, x_2i+1)
    # freq shape [end, dim//2], Q/K维度[end, dim] 用torch.cat([cos,cos],dim=-1)来实现，dim=-1在最后一个维度拼接
    # attn_factor 对 cos/sin做一个整体的缩放
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 将位置信息真正注入 Query 和 Key 向量
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # 后半部分取负并移到前面，前半部分不变移到后面
        # x.shape[-1] // 2: 保留后半部分，-x每个元素取负，并放到前半部分。 最后在最后一维进行拼接
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # 在第一维插入一个维度，用来对齐head的维度
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


# 把key/value 张量里的每个注意力头（head）按 n_rep 次数重复复制，生成更多的 key/value head，以便与 query head 对齐或进行广播计算
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # batch_size一次输入有多少个样本, sequence_length序列长度，也就是每个样本的 token 数量
    # num_key_value_heads key/value 注意力头的数量，head_dim每个头的维度,每个 head 内部的向量长度
    bs, slen, num_key_value_heads, head_dim = x.shape
    # 不需要复制，直接返回x
    if n_rep == 1:
        return x
    # 把 key/value 张量的注意力头按 n_rep 次数复制
    # x shape [bs, slen, num_key_value_heads, head_dim] expand扩展成 [bs, slen, num_key_value_heads, n_rep, head_dim]
    # 再reshape成 [bs, slen, num_key_value_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

# GQA、RoPE、KVcache
class Attention(nn.Module):
    def __init__(self, args: Config):
        super().__init__()
        # 如果没有指定KV头数，就默认和query head一样
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 保证 query head 能整除 key/value head，这样每个 key/value head 可以被几个 query head 共享
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads # attention head数
        self.n_local_kv_heads = self.num_key_value_heads # kv head数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # K/V头被几个Q头共享
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个 attention head 的维度
        # 分别用线性层生成 Q、K、V，Q针对所有的query head，K/V针对少量key/value head
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 最后将多头注意力的输出合并回原来的隐藏维度，这里用的是query head
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout) # 创建用于注意力权重的 dropout 层。
        self.resid_dropout = nn.Dropout(args.dropout) # 创建用于残差连接之后的 dropout 层
        self.dropout = args.dropout # 存到实例self.dropout
        # 判断Flash是否可用，（pytorch>=2.0支持高效注意力）
        # torch.nn.functional.scaled_dot_product_attention，它会自动选择最高效的注意力实现
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                # 接收旋转位置编码
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                # KV缓存
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape # x.shape [batch_size, seq_len, hidden_size]
        # xq (bsz, seq_len, num_query_heads * head_dim) -> (bsz, seq_len, n_local_heads, head_dim)
        # xk (bsz, seq_len, num_kv_heads * head_dim) -> (bsz, seq_len, n_local_kv_heads, head_dim)
        # xv (bsz, seq_len, num_kv_heads * head_dim) -> (bsz, seq_len, n_local_kv_heads, head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆成多头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 从position_embeddings中接收cos和sin
        cos, sin = position_embeddings
        # 位置信息注入QK向量    模型支持的最大序列长度是 max_seq_len，但当前输入的实际序列长度是 seq_len，可能小于最大长度；
        # 所以只需要取前 seq_len 个位置对应的 cos 和 sin 值，与当前 batch 的 xq、xk 对齐
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            # 把之前的kv value在dim=1上进行拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None # 如果use_cache=True，则保存当前时间步的key和value组成新缓存past_kv

        xq, xk, xv = (
            # KV复制到和Q头数相同，最终QKV维度完全相同。 1、2维度交换，qkv维度维度调整为(batch_size, num_heads, seq_len, head_dim)
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        # 如果 PyTorch 支持 flash attention，并且序列长度大于 1，则用高效实现，Flash Attention不支持复杂mask
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 高效attention操作，只有训练时启用dropout_p，is_causal=True自动加因果mask
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # QK^T / sqrt(d)
            scores = scores + torch.triu(
                # torch.triu(···,diagonal=1)上三角全部是-inf，表示token不能看未来 scores+mask 屏蔽上三角的值
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # 返回的是一个二维矩阵，所以需要升维两次

            if attention_mask is not None:
                # attention_mask shape (bsz, seq_len) -> (bsz, 1, 1, seq_len)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 有效位置从1变成0，无效位置从0变成-10^9
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask # 和无效位置相加后变成-inf(-10^9可以当作负无穷)

            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # 得到注意力权重后转回原dtype
            scores = self.attn_dropout(scores) # 正则化
            output = scores @ xv # 最后乘上V
        # 将最后两个维度合并回来
        # (bsz, n_heads, seq_len, head_dim) -> (bsz, seq_len, n_heads, head_dim) -> (bsz, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output)) # 输出投影，再使用残差链接，得到output
        return output, past_kv

# SwiGLU前馈神经层
class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3) # 8/3是SwiGLU经验公式，LLaMA里面就这样用
            # 对齐到64，提高矩阵乘法的效率，是工业级模型的标准操作
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # gate决定哪些维度该通过，up提供真正的信息内容
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) # 生成门控信号，使用激活函数
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 生成被调制的信息，不加激活，保留线性表达能力
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False) # 压回hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] # 根据config.hidden_act决定使用什么激活函数，这里前面定义了是silu

    def forward(self, x):
        # 门控相乘 gate、up size (bsz, seq_len, intermediate_size)
        # down size (bs, seq_len, intermediate_size) -> (bs, seq_len, hidden_size)
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# 一个 Block = 自注意力 + 前馈网络 + 两次 RMSNorm + 残差连接
class Block(nn.Module):
    def __init__(self, layer_id: int, config: Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # 多头注意力的头数
        self.hidden_size = config.hidden_size # 总空间维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config) # GQA + RoPE + KV cache

        # 在Attention前和MLP前，各放一个RMSNorm。对每个token的hidden_size维向量，做RMSNorm，最后eps为了防止分母为0设置的稳定项
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) # SwiGLU前馈网络

    # hidden_states (bsz, seq_len, hidden_size);position_embeddings (cos, sin);past_key_value KV cache 推理时用
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 保存输入，用于残差链接
        # 1.先 RMSNorm 2.把归一化后的结果送入 Attention 3. Attention 输出新的 hidden_states，更新后的 KV cache
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual # 第一条残差 x <- x + Attention(RMSNorm(x)),把注意力层学到的新信息加到旧信息上
        # 第二条残差FFN 1.再做一次 RMSNorm 2.送入 SwiGLU FFN 3.输出和输入再做一次残差相加 x <- x + FFN(RMSNorm(x))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) # 修正数据分布
        return hidden_states, present_key_value # hidden_state本层输出，present_key_value供下一步生成使用


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size # 词表大小
        self.num_hidden_layers = config.num_hidden_layers # 模型有多少层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # 把离散token_id 变成连续语义向量
        self.dropout = nn.Dropout(config.dropout) # 防止过拟合
        # Block × num_hidden_layers，一个Block就是一个完整的transformer层，这个操作把各个层给造出来
        self.layers = nn.ModuleList([Block(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # dim：每个attention head的维度；end：最大位置长度；rope_base：控制不同维度旋转频率的衰减速度, rope_scaling：长上下文扩展策略
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # RoPE的cos和sin注册成“非参数张量缓冲区”，它们会随模型一起to.(device)但不参与训练，默认不保存进checkpoint
        # （声明这是参数的一部分，但不是参数）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # Decoder-only Transformer的完整前向传播逻辑
    # input_ids:输入的token_id; past_value-keys:每一层保存的KV cache，用于下一步增量推理；use_cache:是否返回present用于下一步增量推理
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        # 先判断 past_key_values 是否是 HuggingFace 旧接口（有 .layers 属性），如果是就丢掉
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) # 如果为空，则初始化为[None] * num_layers
        # 初始位置，如果是增量推理(past_key_values 非 None),已经生成了past_len个token，新token的position从past_len开始,否则从0开始
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids)) # 把token_id -> hidden_vector (batch,seq,hidden_size)

        # 生成当前位置的RoPE embedding
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)


        return hidden_states, presents


class ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Config

    def __init__(self, config: Config = None):
        self.config = config or Config()
        super().__init__(self.config)
        self.model = Model(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        return output
