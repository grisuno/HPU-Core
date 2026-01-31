import torch
import numpy as np
import argparse

def analize_checkpoint(path):

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    val_acc = ckpt.get('val_acc', 0)
    metrics = ckpt.get('metrics', {})
    delta = metrics.get('delta', 'N/A')
    config = ckpt.get('config', {})
    q_lambda = config.get('quant_lambda', 0)
    history = config.get('lambda_history', [])
    capas = list(ckpt['model_state_dict'].keys())

    print(f"\n--- Analysis of {path} ---")
    print(f"Real Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Accuracy (ValAcc): {val_acc:.4f}")
    print(f"Delta δ: {delta}")
    print(f"Lambda λ: {q_lambda:.2e}")
    print(f"Lambda Growth: {len(history)}")
    print(f"Layers: {len(capas)}")

    return ckpt

path_old = 'crystal_checkpoints/crystal_epoch_6309_delta_0.3691_20260128_172927_741780.pth'
path_new = 'crystal_checkpoints/crystal_epoch_10970_delta_0.2403_20260131_153555_841521.pth' 

try:
    parser = argparse.ArgumentParser(description="Process Pth checkpoints with fallback defaults.")
    parser.add_argument("--file1", default=path_old, help="Path to the first Pth file (fallback: %(default)s)")
    parser.add_argument("--file2", default=path_new, help="Path to the second Pth file (fallback: %(default)s)")

    args = parser.parse_args()


    data_old = analize_checkpoint(args.file1)
    data_new = analize_checkpoint(args.file2)
    print("\n" + "="*30)
    print(f"Better Result: {float(data_old['metrics']['delta']) - float(data_new['metrics']['delta']):.4f} in delta δ.")
    print("="*30)
    
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
