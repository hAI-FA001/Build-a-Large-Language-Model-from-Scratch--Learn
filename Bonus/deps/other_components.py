import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.emb_dim = emb_dim
        # only 1 parameter
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()
    
    def forward(self, x):
        # RMS -> Root Mean Square
        # x^2 -> mean -> root
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        # normalize input by this
        return (x_normed * self.weight).to(dtype=x.dtype)

class RMSNorm_Qwen(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()

        self.eps = eps
        # Llama had only 1 parameter
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)  # for Qwen
        
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)

        # new
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        
        return norm_x.to(input_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward_SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # cfg['dtype'] will allow loading in lower precision format
        self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc3 = nn.Linear(cfg['hidden_dim'], cfg['emb_dim'], dtype=cfg['dtype'], bias=False)
        self.silu = SiLU()
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_len=4096, freq_config=None, dtype=None):
    assert head_dim % 2 == 0, 'Head dim must be even'

    p = torch.arange(0, head_dim, 2, dtype=dtype)
    p = p[:head_dim//2].float()
    p = p / head_dim
    inv_freq = 1.0 / theta_base**p

    # for Llama 3.1 and 3.2
    if freq_config is not None:
        low_freq_wavelen = freq_config['original_context_len'] \
                            / freq_config['low_freq_factor']
        high_freq_wavelen = freq_config['original_context_len'] \
                            / freq_config['high_freq_factor']
        
        wavelen = 2*torch.pi / inv_freq
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / freq_config['factor'],
            inv_freq,
        )

        smooth_factor = (freq_config['original_context_len'] \
                        / wavelen - freq_config['low_freq_factor']) \
                        /(freq_config['high_freq_factor'] - freq_config['low_freq_factor'])
        smoothed_inv_freq = (1-smooth_factor) * (inv_freq / freq_config['factor']) \
                        + smooth_factor * inv_freq
        
        is_med_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_med_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    
    positions = torch.arange(context_len, dtype=dtype)
    angles = positions[:,None] * inv_freq[None,:]
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, 'Head dimension must be even'

    # split into 2 halves
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]

    # (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x*cos) + (rotated*sin)

    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention_Qwen(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups,
                    head_dim=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, 'num_heads must be divisible by num_kv_groups'

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, 'd_in must be divisible by num_heads if head_dim is not provided'
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        # if head_dim not provided then:
        #   d_out = num_heads * head_dim = num_heads * (d_in / num_heads) = d_in
        #   so, d_out = d_in
        self.d_out = num_heads*head_dim

        self.W_q = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # if head_dim not provided, then:
        #   num_kv_groups * head_dim = num_kv_groups * d_in / num_heads
        #       = (num_kv_groups / num_heads) * d_in
        #       = (1 / group_size) * d_in = d_in / group_size
        # in GQA, K and V are repeated "group_size" times, so:
        #       = group_size * ((1 / group_size) * d_in)
        #       = d_in
        self.W_k = nn.Linear(d_in, num_kv_groups*head_dim, bias=False, dtype=dtype)
        self.W_v = nn.Linear(d_in, num_kv_groups*head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        # new, QK-Norm
        self.q_norm = RMSNorm_Qwen(head_dim, eps=1e-6)
        self.k_norm = RMSNorm_Qwen(head_dim, eps=1e-6)
    
    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_q(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_k(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_v(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # "un-flatten" last dims
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # new
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = compute_rope(queries, cos, sin)
        keys = compute_rope(keys, cos, sin)

        # (..., head_dim, num_kv_groups) -> (..., head_dim, num_heads)
        #   num_heads / num_kv_groups = group_size
        #   => num_heads = num_kv_groups * group_size
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2,3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1,2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)