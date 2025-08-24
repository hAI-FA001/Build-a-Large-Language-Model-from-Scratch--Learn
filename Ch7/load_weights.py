import numpy as np
import torch
from gpt_model import GPTModel

# utility to check if 2 tensors have same shape
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f'Shape Mismatch: Left {left.shape}, Right {right.shape}')
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):
        # Weights
        q_w, k_w, v_w = np.split(
            params['blocks'][b]['attn']['c_attn']['w'],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].attn.W_q.weight = assign(
            gpt.trf_blocks[b].attn.W_q.weight,
            q_w.T
        )
        
        gpt.trf_blocks[b].attn.W_k.weight = assign(
            gpt.trf_blocks[b].attn.W_k.weight,
            k_w.T
        )
        
        gpt.trf_blocks[b].attn.W_v.weight = assign(
            gpt.trf_blocks[b].attn.W_v.weight,
            v_w.T
        )

        # Biases
        q_b, k_b, v_b = np.split(
            params['blocks'][b]['attn']['c_attn']['b'],
            3,
            axis=-1
        )

        gpt.trf_blocks[b].attn.W_q.bias = assign(
            gpt.trf_blocks[b].attn.W_q.bias,
            q_b.T
        )
        
        gpt.trf_blocks[b].attn.W_k.bias = assign(
            gpt.trf_blocks[b].attn.W_k.bias,
            k_b.T
        )
        
        gpt.trf_blocks[b].attn.W_v.bias = assign(
            gpt.trf_blocks[b].attn.W_v.bias,
            v_b.T
        )

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params['blocks'][b]['attn']['c_proj']['b']
        )

        # Feed Forward Network
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params['blocks'][b]['mlp']['c_fc']['w'].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params['blocks'][b]['mlp']['c_fc']['b']
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params['blocks'][b]['mlp']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params['blocks'][b]['mlp']['c_proj']['b']
        )

        # Layer Norms
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params['blocks'][b]['ln_1']['g']
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params['blocks'][b]['ln_1']['b']
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params['blocks'][b]['ln_2']['g']
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params['blocks'][b]['ln_2']['b']
        )
    
    gpt.final_norm.scale = assign(
        gpt.final_norm.scale,
        params['g']
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift,
        params['b']
    )
    gpt.out_head.weight = assign(
        gpt.out_head.weight,
        params['wte']
    )