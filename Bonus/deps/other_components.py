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

