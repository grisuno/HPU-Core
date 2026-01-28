#!/usr/bin/env python3
"""
Calcula la constante de Planck efectiva (ħ) desde checkpoints de HPU.

Física del cálculo:
- Trata λ como potencial químico de confinamiento
- δ como longitud de penetración cuántica  
- Deriva ħ desde relación de incertidumbre generalizada
"""

import torch
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any


H_BAR_SI = 1.054571817e-34  # J·s
C = 299792458  # m/s
G = 6.67430e-11  # m³/kg·s²
M_SUN = 1.98847e30  # kg


class HBarCalculator:
    """Calcula ħ efectiva desde checkpoint HPU usando física realista."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.checkpoint = torch.load(
            checkpoint_path, 
            map_location=device, 
            weights_only=False
        )

        self.metrics = self.checkpoint.get('metrics', {})
        self.epoch = self.checkpoint.get('epoch', 0)

        self.lambda_val = self.metrics.get('quant_lambda', 
                          self.metrics.get('lambda', 0.5))
        self.delta = self.metrics.get('delta', 1.0)
        self.alpha = self.metrics.get('alpha', 0.0)
        self.mse = self.metrics.get('val_mse', 
                   self.metrics.get('mse', 1.0))
        self.val_acc = self.checkpoint.get('val_acc', 0.0)

        config = self.checkpoint.get('config', {})
        self.lr = config.get('lr', 0.001)
        self.grid_size = config.get('grid_size', 16)

    def calculate_all(self) -> Dict[str, Any]:
        """Ejecuta todos los cálculos de ħ."""

        # MÉTODO 1: Incertidumbre generalizada
        # ħ ~ 2·δ²·λ (de Δx·Δp ≥ ħ/2 con Δx=δ, Δp~λ·δ)
        h_bar_uncertainty = 2 * self.delta**2 * self.lambda_val

        # MÉTODO 2: Cuantización de acción
        # ω = √λ (frecuencia natural del confinamiento armónico)
        omega = np.sqrt(self.lambda_val) if self.lambda_val > 0 else 1.0
        period = 2 * np.pi / omega if omega > 0 else 1.0

        T = self.mse  # Energía cinética
        V = self.lambda_val * self.delta**2  # Energía potencial
        L = T - V  # Lagrangiano
        action = abs(L) * period
        h_bar_action = action

        # MÉTODO 3: Conductancia cuántica (Efecto Hall análogo)
        # G = I/V = acc/mse, ħ ~ 1/G
        if self.mse > 0:
            conductance = self.val_acc / self.mse
            h_bar_conductance = 1.0 / conductance
        else:
            h_bar_conductance = 0.0

        # MÉTODO 4: Entropía de información
        # Asumiendo ~31 modos efectivos del análisis espectral
        N_eff = 31
        I = np.log2(N_eff) if N_eff > 1 else 1.0
        E_total = T + V
        energy_per_bit = E_total / I if I > 0 else 0
        h_bar_information = energy_per_bit * period

        # CONSOLIDACIÓN: Promedio ponderado
        # Prioridad alta al método de incertidumbre en régimen λ→∞
        if self.lambda_val > 1e30:
            w1, w2, w3, w4 = 0.6, 0.25, 0.1, 0.05  # Ultra-strong
        elif self.lambda_val > 1e10:
            w1, w2, w3, w4 = 0.5, 0.3, 0.15, 0.05  # Strong
        else:
            w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25  # Weak

        total_w = w1 + w2 + w3 + w4
        h_bar_final = (w1*h_bar_uncertainty + w2*h_bar_action + 
                      w3*h_bar_conductance + w4*h_bar_information) / total_w

        # Adimensionalización
        h_bar_scale = self.mse * period
        h_bar_dimless = h_bar_final / h_bar_scale if h_bar_scale > 0 else 0

        # CONSTANTES DERIVADAS
        v_light = C * (h_bar_final / H_BAR_SI) if H_BAR_SI > 0 else 0
        m_planck = np.sqrt(h_bar_final * v_light / G) if G > 0 and v_light > 0 else 0
        l_planck = np.sqrt(h_bar_final * G / v_light**3) if v_light > 0 else 0
        t_planck = l_planck / v_light if v_light > 0 else 0

        k_B = 1.380649e-23
        T_planck = m_planck * v_light**2 / k_B if k_B > 0 else 0

        # COMPARACIÓN
        ratio_h = h_bar_final / H_BAR_SI if H_BAR_SI > 0 else 0
        orders = np.log10(ratio_h) if ratio_h > 0 else 0

        # RÉGIMEN
        if self.lambda_val > 1e30:
            regime = "ULTRA-STRONG CONFINEMENT"
        elif self.lambda_val > 1e10:
            regime = "STRONG CONFINEMENT"
        elif self.lambda_val > 1.0:
            regime = "WEAK CONFINEMENT"
        else:
            regime = "UNCONSTRAINED"

        return {
            "metadata": {
                "checkpoint": self.checkpoint_path,
                "epoch": self.epoch,
                "timestamp": datetime.now().isoformat(),
                "calculation_version": "1.0.0"
            },
            "inputs": {
                "lambda": self.lambda_val,
                "delta": self.delta,
                "alpha": self.alpha,
                "mse": self.mse,
                "val_acc": self.val_acc,
                "grid_size": self.grid_size,
                "learning_rate": self.lr
            },
            "h_bar": {
                "value": float(h_bar_final),
                "dimensionless": float(h_bar_dimless),
                "regime": regime,
                "methods": {
                    "uncertainty": float(h_bar_uncertainty),
                    "action": float(h_bar_action),
                    "conductance": float(h_bar_conductance),
                    "information": float(h_bar_information)
                },
                "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4}
            },
            "derived_constants": {
                "c_effective_m_s": float(v_light),
                "m_planck_kg": float(m_planck),
                "l_planck_m": float(l_planck),
                "t_planck_s": float(t_planck),
                "T_planck_K": float(T_planck)
            },
            "comparison_universe": {
                "h_bar_ratio": float(ratio_h),
                "orders_of_magnitude": float(orders),
                "m_planck_vs_sun": float(m_planck / M_SUN) if M_SUN > 0 else 0
            }
        }

    def print_report(self, results: Dict):
        """Imprime reporte formateado."""
        h = results["h_bar"]
        d = results["derived_constants"]
        c = results["comparison_universe"]

        print(f"DeepLeaning PLANCK - EPOCH {results['metadata']['epoch']}")
        

        print(f"\n[INPUTS]")
        print(f"  λ = {results['inputs']['lambda']:.6e}")
        print(f"  δ = {results['inputs']['delta']:.6f}")
        print(f"  MSE = {results['inputs']['mse']:.6e}")
        print(f"  Acc = {results['inputs']['val_acc']:.4f}")

        print(f"\n[RESULTS]")
        print(f"  ħ = {h['value']:.6e} [sistema]")
        print(f"  ħ_adimensional = {h['dimensionless']:.6e}")
        print(f"  Régimen: {h['regime']}")

        print(f"\n[METHODS]")
        for name, val in h['methods'].items():
            print(f"  {name:15s}: {val:.6e}")
        print(f"  Pesos: {h['weights']}")

        print(f"\n[DEREIVATED CONSTANTS]")
        print(f"  c_eff = {d['c_effective_m_s']:.6e} m/s")
        print(f"  m_P = {d['m_planck_kg']:.6e} kg")
        print(f"  l_P = {d['l_planck_m']:.6e} m")
        print(f"  t_P = {d['t_planck_s']:.6e} s")
        print(f"  T_P = {d['T_planck_K']:.6e} K")

        print(f"\n[VS UNIVERSE]")
        print(f"  ħ_ratio = {c['h_bar_ratio']:.6e}")
        print(f"  Δórdenes = {c['orders_of_magnitude']:+.1f}")
        print(f"  m_P/M_☉ = {c['m_planck_vs_sun']:.6e}")

        if c['orders_of_magnitude'] > 0:
            print(f"\n  → ħ is {c['orders_of_magnitude']:.0f} orders more than real fisics")




def main():
    parser = argparse.ArgumentParser(
        description='Calc ħ efective from checkpoint HPU'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default='crystal_checkpoints/latest.pth',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='JSON output file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error:  {args.checkpoint}")
        return 1

    calc = HBarCalculator(args.checkpoint)
    results = calc.calculate_all()
    calc.print_report(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved in: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
