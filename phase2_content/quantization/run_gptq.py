import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase1_content.nanoGPT_annotated.nano_gpt import BiLanguageModel, config, get_batch
from ptq import PTQuantizer
from gptq import apply_gptq




def print_tensor_stats(name, tensor: torch.Tensor):
    print(f"{name} stats:")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min:   {tensor.min().item():.6f}")
    print(f"  max:   {tensor.max().item():.6f}")
    print(f"  mean:  {tensor.mean().item():.6f}")
    print()


def main():

    # 1. Load model
    model = BiLanguageModel(config)
    model.load_state_dict(
        torch.load(
            "phase1_content/nanoGPT_annotated/model.pt",
            map_location="cpu"
        )
    )
    model.eval()

    # 2. Generate calibration data
    calibration_data = []
    for _ in range(10):
        x, y = get_batch("train", config)
        calibration_data.append((x, y))

    # 3. Instantiate quantizer
    quantizer = PTQuantizer()

    # 4. Stats before quantization (lm_head)
    print_tensor_stats("lm_head.weight BEFORE", model.lm_head.weight.data)

    # 5. Apply GPTQ
    apply_gptq(
        model=model,
        calibration_data=calibration_data,
        quantizer=quantizer,
        damping=0.01
    )

    # 6. Stats after quantization
    print_tensor_stats("lm_head.weight AFTER", model.lm_head.weight.data)

    # 7. Done
    print("Done.")


if __name__ == "__main__":
    main()