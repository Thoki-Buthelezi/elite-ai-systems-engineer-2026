import torch
import torch.nn as nn
from typing import Dict, List


def collect_inputs(
    model: nn.Module,
    calibration_data
) -> Dict[nn.Linear, torch.Tensor]:
    """
    Collect the input activations for every Linear layer in the model.

    Args:
        model: The model to collect activations from.
        calibration_data: Iterable of (x, y) calibration batches.

    Returns:
        Dictionary mapping each nn.Linear layer to its collected input
        activations.
    """

    collected_inputs: Dict[nn.Linear, List[torch.Tensor]] = {}
    hooks = []

    def hook_fn(layer):
        def hook(module, inputs, output):
            # Linear layers receive a single input tensor.
            x = inputs[0].detach().cpu()

            if layer not in collected_inputs:
                collected_inputs[layer] = []

            collected_inputs[layer].append(x)

        return hook

    # Register hooks on every Linear layer.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(module)))

    model.eval()

    try:
        with torch.no_grad():
            for x, y in calibration_data:
                model(x)
    finally:
        # Always remove hooks, even if a forward pass fails.
        for hook in hooks:
            hook.remove()

    # Concatenate collected activations.
    final_inputs: Dict[nn.Linear, torch.Tensor] = {}

    for layer, tensors in collected_inputs.items():
        final_inputs[layer] = torch.cat(tensors, dim=0)

    return final_inputs

def gptq_quantize_layer(
    layer: nn.Linear,
    inputs: torch.Tensor,
    quantizer,
    damping: float = 0.01
) -> None:
    """
    Quantize a single nn.Linear layer in-place using GPTQ.
    """

    W = layer.weight.data.clone()  # (out_features, in_features)

    # flatten activations if needed
    if inputs.dim() > 2:
        inputs = inputs.view(-1, inputs.shape[-1])

    X = inputs  # (N, in_features)

    # 1. Hessian approximation
    H = 2.0 * (X.T @ X)

    # 2. Damping
    diag_mean = torch.mean(torch.diag(H))
    H += damping * diag_mean * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    # 3. Cholesky inverse Hessian
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)

    in_features = W.shape[1]

    W_q = torch.zeros_like(W)

    for j in range(in_features):

        # 4. Quantize column j
        w_j = W[:, j]

        w_q_j, scale = quantizer.quantize_symmetric(w_j)
        w_q_j = quantizer.dequantize_symmetric(w_q_j, scale)

        W_q[:, j] = w_q_j

        # 5. Quantization error
        err_j = w_j - w_q_j  # (out_features,)

        # 6. Propagate error to remaining columns
        if j < in_features - 1:

            Hjj = H_inv[j, j]
            if Hjj.abs() < 1e-8:
                Hjj = Hjj + 1e-8

            scale_vec = H_inv[j, j + 1:] / Hjj  # (in_features - j - 1,)

            err_scaled = err_j.unsqueeze(1) * scale_vec.unsqueeze(0)

            W[:, j + 1:] -= err_scaled

    # 7. Write back
    layer.weight.data.copy_(W_q)

def apply_gptq(
    model: nn.Module,
    calibration_data,
    quantizer,
    damping: float = 0.01
) -> None:
    """
    Apply GPTQ quantization to all nn.Linear layers in the model in-place.
    """

    # 1. collect activations per linear layer
    layer_inputs = collect_inputs(model, calibration_data)

    # 2. quantize each layer
    for i, (layer, inputs) in enumerate(layer_inputs.items()):
        print(f"[GPTQ] Quantizing layer {i + 1}/{len(layer_inputs)}: {layer}")

        gptq_quantize_layer(
            layer=layer,
            inputs=inputs,
            quantizer=quantizer,
            damping=damping
        )