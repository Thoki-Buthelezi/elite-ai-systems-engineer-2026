import torch

class PTQuantizer:

    def quantize_symmetric(self, tensor: torch.Tensor) -> tuple:
        """
        Symmetric int8 quantization.
        Range: [-127, 127]
        Returns:
            (q_tensor, scale)
        """

        max_abs = torch.max(torch.abs(tensor))

        # Avoid division by zero
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / 127.0

        q_tensor = torch.round(tensor / scale)
        q_tensor = torch.clamp(q_tensor, -127, 127).to(torch.int8)

        return q_tensor, scale

    def quantize_asymmetric(self, tensor: torch.Tensor) -> tuple:
        """
            Asymmetric int8 quantization.
            Range: [-128, 127]
            Returns:
            (q_tensor, scale, zero_point)
        """

        qmin = -128
        qmax = 127

        x_min = tensor.min()
        x_max = tensor.max()

        # Edge case: constant tensor
        if x_max == x_min:
            scale = 1.0
            zero_point = 0
            q_tensor = torch.zeros_like(tensor, dtype=torch.int8)
            return q_tensor, scale, zero_point

        scale = (x_max - x_min) / float(qmax - qmin)

        # corrected zero-point form (as derived)
        zero_point = qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int32)

        # corrected quantization ordering
        q_tensor = torch.round(tensor / scale) + zero_point
        q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)

        return q_tensor, scale, zero_point
    
    
    def dequantize_symmetric(self, q_tensor, scale) -> torch.Tensor:
        """
        x ≈ q * scale
        """
        return q_tensor.float() * scale

    def dequantize_asymmetric(self, q_tensor, scale, zero_point) -> torch.Tensor:
        """
        x ≈ scale * (q - zero_point)
        """
        return scale * (q_tensor.float() - zero_point)
    

if __name__ == "__main__":
    torch.manual_seed(1024)
    size = 93484

    q = PTQuantizer()

    x = torch.randn(size, dtype=torch.float)

    x_quantised_symmetric, scale_symmetric = q.quantize_symmetric(x)
    x_quantised_asymmetric, scale_asymmetric , zero_point = q.quantize_asymmetric(x)

    x_dequantised_symmetric =  q.dequantize_symmetric(x_quantised_symmetric, scale_symmetric)
    x_dequantised_asymmetric = q.dequantize_asymmetric(x_quantised_asymmetric, scale_asymmetric, zero_point)

    max_err_asymmetric = (x - x_dequantised_asymmetric).abs().max().item()
    max_err_symmetric = (x - x_dequantised_symmetric).abs().max().item()
    print(f"Asymmetric max error: {max_err_asymmetric:.6f}")
    print(f"Symmetric max error: {max_err_symmetric:.6f}")

