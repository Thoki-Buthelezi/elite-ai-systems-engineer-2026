import torch
torch.manual_seed(1999)

def speculative_step(model_p, model_q, prefix, gamma):

    # 1. Draft. Mq generates gamma tokens autogressively
    draft_tokens = []
    draft_probs = [] # q(x) at each position, need these to reject/accpet 
    current = prefix

    for _ in range(gamma):
        logits, _ = model_q(current)
        q_dist = torch.softmax(logits[:, -1, :], dim=-1)
        x = torch.multinomial(q_dist, num_samples=1) # sample from q(x)
        draft_probs.append(q_dist)
        draft_tokens.append(x)
        current = torch.cat([current, x], dim=1)

    # 2. Verify: ONE forward pass of Mp over prefix + all draft tokens
    full_seq = torch.cat([prefix] + draft_tokens, dim=1)
    logits_p, _ = model_p(full_seq)

    # 3. Sequential accept/reject, left to right
    accepted = []
    for i in range(gamma):
        p_dist = torch.softmax(logits_p[:, prefix.shape[1] - 1 + i, :], dim=-1)
        q_dist = draft_probs[i]
        x = draft_tokens[i]
        x_id = x.item()
        r = torch.rand(1).item()
        accept_prob = min(1.0, (p_dist[0, x_id] / q_dist[0, x_id]).item())

        # #logging results will remove after testing
        # print(f"p(x) {p_dist[0, x_id]}")
        # print(f"q(x) {q_dist[0, x_id]}")
        # print(f"accepted_prob {accept_prob}")

        if r < accept_prob:
            # print(f"r->{r} , decision->accepted")
            accepted.append(x)
        else:
            print(f"r->{r} , decision->rejected")
            residual = torch.clamp(p_dist - q_dist, min=0) # min[0, p(x) - q(x)]
            residual = residual / residual.sum() # normalize
            x_new = torch.multinomial(residual, num_samples=1)
            accepted.append(x_new)
            # print(f"final len of accepted: {len(accepted)}")
            return accepted
        
    # all gamma accepted, sample from p_dist at possition gamma
    bonus_dist = torch.softmax(logits_p[:, prefix.shape[1] - 1 + gamma, :], dim=-1)
    bonus = torch.multinomial(bonus_dist, num_samples=1)
    accepted.append(bonus)
    # print(f"final len of accepted: {len(accepted)}")
    return accepted
        