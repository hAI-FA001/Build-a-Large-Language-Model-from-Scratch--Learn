import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        # take last time step
        logits = logits[:,-1,:]
        # (B, V)
        probas = torch.softmax(logits, dim=-1)
        # (B, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # (B, N) & (B, 1) -> (B, N+1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx