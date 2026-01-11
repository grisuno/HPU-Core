#!/usr/bin/env python3
"""
Zero-shot spectral expansion of grokked Hamiltonian operator.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import toml
from typing import Dict, List
from main_fast import SimpleHamiltonianNet, FastDataset, compute_local_complexity, compute_superposition
from typing import Tuple
import time
import json


def load_config(toml_path: str = "config.toml") -> Dict:
    with open(toml_path, "r") as f:
        return toml.load(f)


def expand_spectral_weights(
    kernel_real: torch.Tensor,
    kernel_imag: torch.Tensor,
    target_size: int,
    source_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand spectral kernels via zero-padding in frequency domain."""
    
    pad_h = (target_size // 2 + 1) - kernel_real.size(-2)
    pad_w = target_size - kernel_real.size(-1)

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size must be >= source size")

    kernel_real_exp = F.pad(kernel_real, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    kernel_imag_exp = F.pad(kernel_imag, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    return kernel_real_exp, kernel_imag_exp


def expand_model(
    model: SimpleHamiltonianNet,
    target_resolution: int,
    source_resolution: int = 16
) -> SimpleHamiltonianNet:
    """Create a new model with expanded spectral weights."""
    new_model = SimpleHamiltonianNet(
        grid_size=target_resolution,
        hidden_dim=model.input_proj.out_channels,
        num_spectral_layers=len(model.spectral_layers)
    )

    
    new_model.input_proj.load_state_dict(model.input_proj.state_dict())
    new_model.output_proj.load_state_dict(model.output_proj.state_dict())

    
    for i, layer in enumerate(model.spectral_layers):
        kernel_real = layer.kernel_real.data
        kernel_imag = layer.kernel_imag.data

        kernel_real_exp, kernel_imag_exp = expand_spectral_weights(
            kernel_real, kernel_imag, target_resolution, source_resolution
        )

        new_model.spectral_layers[i].kernel_real.data = kernel_real_exp
        new_model.spectral_layers[i].kernel_imag.data = kernel_imag_exp

    return new_model


def evaluate_model(model: SimpleHamiltonianNet, resolution: int, device: torch.device) -> Dict:
    """Evaluate expanded model on synthetic data."""
    model.eval()
    dataset = FastDataset(num_samples=50, grid_size=resolution, time_steps=2)
    val_x, val_y = dataset.get_val_batch()
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    with torch.no_grad():
        outputs = model(val_x)
        mse = F.mse_loss(outputs, val_y).item()
        mse_per_sample = ((outputs - val_y) ** 2).mean(dim=(1, 2))
        acc = (mse_per_sample < 0.05).float().mean().item()

    
    lc_values = []
    sp_values = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param[:min(param.size(0), 256), :min(param.size(1), 256)]
            lc = compute_local_complexity(w)
            sp = compute_superposition(w)
            lc_values.append(lc)
            sp_values.append(sp)

    lc = np.mean(lc_values) if lc_values else 0.0
    sp = np.mean(sp_values) if sp_values else 0.0

    return {"mse": mse, "acc": acc, "lc": lc, "sp": sp}


def main():
    config = load_config()
    source_res = config["model"]["source_resolution"]
    checkpoint_path = config["model"]["checkpoint_path"]
    target_resolutions = config["target_resolutions"]["resolutions"]
    save_dir = config["output"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]
    model = SimpleHamiltonianNet(
        grid_size=source_res,
        hidden_dim=model_cfg["hidden_dim"],
        num_spectral_layers=model_cfg["num_spectral_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded grokked model from {checkpoint_path} (res={source_res})")

    results = {}
    start_time = time.time()
    for res in target_resolutions:
        print(f"\n[+] Expanding to resolution {res}...")
        try:
            expanded_model = expand_model(model, res, source_res).to(device)
            metrics = evaluate_model(expanded_model, res, device)
            duration = time.time() - start_time
            results[res] = metrics
            results[res]["duration"] = duration
            print(f"[+] Res {res}: MSE={metrics['mse']:.6f}, Acc={metrics['acc']:.4f}, LC={metrics['lc']:.4f}, SP={metrics['sp']:.4f}, Tiempo={duration:.2f}s")
        except Exception as e:
            print(f"[-] Failed at res {res}: {e}")

    
    
    with open(os.path.join(save_dir, "zero_shot_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[*] Results saved to {save_dir}/zero_shot_results.json")


if __name__ == "__main__":
    main()
