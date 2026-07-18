import torch

# Force a case where q(x) <= p(x) so acceptance is guaranteed (accept_prob = 1.0)
# Manually construct a fake p_dist and q_dist to test the accept path in isolation
p_dist = torch.tensor([[0.1, 0.6, 0.3]])
q_dist = torch.tensor([[0.1, 0.4, 0.5]])
x_id = 1  # p(x)=0.6, q(x)=0.4, accept_prob = min(1.0, 0.6/0.4) = 1.0, always accepted

accept_prob = min(1.0, (p_dist[0, x_id] / q_dist[0, x_id]).item())
assert accept_prob == 1.0