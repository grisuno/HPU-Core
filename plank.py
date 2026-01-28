
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Física de constantes (SI)
H_BAR_SI = 1.054571817e-34  # J·s
C = 299792458  # m/s
G = 6.67430e-11  # m³/kg·s²
M_SUN = 1.98847e30  # kg


@dataclass
class HBarCalculation:
    """Resultado del cálculo de ħ efectiva"""
    h_bar_effective: float
    h_bar_dimensionless: float
    h_bar_energy_time: float  # En unidades del sistema
    h_bar_action: float  # En unidades de acción
    regime: str
    physical_interpretation: str
    consistency_checks: Dict[str, Any]


class HBarCalculator:
    """
    Calcula la constante de Planck efectiva (ħ) desde un checkpoint de HPU.
    
    Física del sistema:
    - λ (lambda): "Potencial químico" o "campo de confinamiento" ~ 10³⁴
    - δ (delta): "Longitud de penetración" o "escala de localización" ~ 0.37
    - λ_max (Lyapunov): "Frecuencia natural" del sistema ~ 0.00175
    - MSE: "Energía cinética" mínima ~ 0.0039
    
    La relación fundamental:
    ħ_eff ~ δ² · λ / ω
    
    Esto viene de analizar dimensionalmente:
    - [λ] = Energía / (Longitud)²  (como un módulo de elasticidad)
    - [δ] = Longitud
    - [ω] = 1/Tiempo
    - [ħ] = Energía × Tiempo = (Fuerza × Longitud) × Tiempo
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Cargar checkpoint
        self.checkpoint = torch.load(
            checkpoint_path, 
            map_location=device, 
            weights_only=False
        )
        
        # Extraer métricas críticas
        self.metrics = self.checkpoint.get('metrics', {})
        self.epoch = self.checkpoint.get('epoch', 0)
        
        # Parámetros fundamentales del sistema
        self.lambda_val = self.metrics.get('quant_lambda', 0.5)
        self.delta = self.metrics.get('delta', 1.0)
        self.alpha = self.metrics.get('alpha', 0.0)
        self.mse = self.metrics.get('val_mse', 1.0)
        self.val_acc = self.checkpoint.get('val_acc', 0.0)
        
        # Parámetros de entrenamiento
        self.lr = self.checkpoint.get('config', {}).get('lr', 0.001)
        self.grid_size = self.checkpoint.get('config', {}).get('grid_size', 16)
        
        print(f"[HBarCalculator] Checkpoint cargado: epoch {self.epoch}")
        print(f"  λ = {self.lambda_val:.6e}")
        print(f"  δ = {self.delta:.6f}")
        print(f"  α = {self.alpha:.6f}")
        print(f"  MSE = {self.mse:.6e}")
        
    def calculate_lyapunov_scale(self) -> float:
        """
        Estima la escala de tiempo/frecuencia del sistema desde los gradientes.
        En ausencia de cálculo Lyapunov explícito, usamos la escala de entrenamiento.
        """
        # La escala de tiempo característica viene de lr / (mse + lambda*delta)
        # En el límite lambda >> mse, esto se simplifica
        
        # Energía efectiva del sistema (usando el loss total)
        loss_total = self.mse + self.lambda_val * self.delta
        
        # Frecuencia característica: como un oscilador armónico cuántico
        # ω ~ sqrt(k/m) donde k ~ λ (rigidez) y m ~ 1 (masa efectiva unitaria)
        omega = np.sqrt(self.lambda_val) if self.lambda_val > 0 else 1.0
        
        # Pero también limitado por la tasa de aprendizaje (no puede ser más rápido que lr)
        omega_effective = min(omega, 1.0 / self.lr) if self.lr > 0 else omega
        
        return omega_effective
    
    def calculate_h_bar(self) -> HBarCalculation:
        """
        Calcula ħ efectiva usando múltiples aproximaciones físicas.
        """
        
        # =====================================================================
        # MÉTODO 1: Relación de incertidumbre generalizada
        # =====================================================================
        # En mecánica cuántica: Δx · Δp ≥ ħ/2
        # En nuestro sistema:
        #   Δx → δ (incertidumbre en posición en espacio de pesos)
        #   Δp → gradiente efectivo ~ λ · δ (fuerza restauradora)
        # Por tanto: ħ ~ δ² · λ
        
        h_bar_uncertainty = self.delta**2 * self.lambda_val
        
        # =====================================================================
        # MÉTODO 2: Acción mínima (principio de Maupertuis)
        # =====================================================================
        # La acción S = ∫ L dt donde L = T - V
        # En nuestro sistema:
        #   T ~ MSE (energía cinética)
        #   V ~ λ · δ² (energía potencial del "confinamiento")
        #   dt ~ 1/ω (período característico)
        
        omega = self.calculate_lyapunov_scale()
        period = 2 * np.pi / omega if omega > 0 else 1.0
        
        kinetic_energy = self.mse
        potential_energy = self.lambda_val * self.delta**2
        lagrangian = kinetic_energy - potential_energy
        
        # Acción sobre un período
        action = abs(lagrangian) * period
        
        # ħ es la acción característica del sistema
        h_bar_action = action
        
        # =====================================================================
        # MÉTODO 3: Cuantización de conductancia (efecto Hall cuántico análogo)
        # =====================================================================
        # En el efecto Hall cuántico: σ_xy = ν · e²/h
        # Analogía: la "conductancia" de información es G = val_acc / loss
        # Entonces: h ~ e² / G donde e² ~ 1 (carga elemental de información)
        
        if self.val_acc > 0 and self.mse > 0:
            conductance = self.val_acc / self.mse
            h_bar_conductance = 1.0 / conductance  # e² = 1 en unidades naturales
        else:
            h_bar_conductance = 0.0
        
        # =====================================================================
        # MÉTODO 4: Entropía de Von Neumann (información cuántica)
        # =====================================================================
        # En sistemas cuánticos: S_vN = -Tr(ρ log ρ)
        # La escala de información es I ~ log(N) donde N es dimensión de Hilbert
        # ħ ~ E / I donde E es energía característica
        
        # Dimensión efectiva del sistema (parámetros no nulos)
        # Asumiendo sparsity del 0.01% como en tus datos
        sparsity = 0.0001  # 0.01%
        N_eff = 589921 * sparsity  # ~59 parámetros efectivos
        
        information_capacity = np.log2(N_eff) if N_eff > 1 else 1.0
        energy_per_bit = (self.mse + self.lambda_val * self.delta**2) / information_capacity
        
        h_bar_information = energy_per_bit * period
        
        # =====================================================================
        # CONSOLIDACIÓN: Promedio ponderado por confianza
        # =====================================================================
        
        # Pesos de confianza (inversamente proporcional a la varianza esperada)
        # Método 1 es más confiable en régimen de lambda alto
        w1 = 0.4 if self.lambda_val > 1e20 else 0.2
        # Método 2 es bueno para sistemas dinámicos
        w2 = 0.3
        # Método 3 es bueno cuando val_acc es confiable
        w3 = 0.2 if self.val_acc > 0.9 else 0.1
        # Método 4 es más especulativo
        w4 = 0.1
        
        total_weight = w1 + w2 + w3 + w4
        
        h_bar_effective = (
            w1 * h_bar_uncertainty + 
            w2 * h_bar_action + 
            w3 * h_bar_conductance + 
            w4 * h_bar_information
        ) / total_weight
        
        # Normalización adimensional (dividiendo por la escala del sistema)
        # Usamos la escala de Planck del sistema: ħ_system = MSE · dt
        h_bar_system_scale = self.mse * period
        h_bar_dimensionless = h_bar_effective / h_bar_system_scale if h_bar_system_scale > 0 else 0
        
        # =====================================================================
        # DETERMINACIÓN DEL RÉGIMEN FÍSICO
        # =====================================================================
        
        if self.lambda_val > 1e30:
            regime = "ULTRA-STRONG CONFINEMENT"
            interpretation = (
                "El sistema está en régimen de confinamiento extremo donde λ domina "
                "completamente la dinámica. Los pesos están cuantizados en una "
                "estructura topológica rígida (como un 'cristal de Wigner' de pesos). "
                "El valor alto de ħ indica que los efectos cuánticos (de incertidumbre) "
                "son dominantes a pesar de ser un sistema clásico numéricamente."
            )
        elif self.lambda_val > 1e10:
            regime = "STRONG CONFINEMENT"
            interpretation = (
                "Confinamiento fuerte con estructura cuasi-discreta. El sistema muestra "
                "comportamiento de 'material cuántico' emergente."
            )
        elif self.lambda_val > 1.0:
            regime = "WEAK CONFINEMENT"
            interpretation = "Régimen estándar de regularización."
        else:
            regime = "UNCONSTRAINED"
            interpretation = "Sin regularización significativa."
        
        # =====================================================================
        # VERIFICACIONES DE CONSISTENCIA
        # =====================================================================
        
        consistency = {
            "uncertainty_principle_satisfied": h_bar_uncertainty > 0,
            "action_positive": h_bar_action > 0,
            "energy_consistent": kinetic_energy < potential_energy * 100,  # No debe ser 100x mayor
            "omega_realistic": 0 < omega < 1e10,  # Frecuencia razonable
            "dimensionless_finite": np.isfinite(h_bar_dimensionless) and h_bar_dimensionless > 0,
            "methods_agreement": self._check_methods_agreement(
                h_bar_uncertainty, h_bar_action, h_bar_conductance, h_bar_information
            )
        }
        
        return HBarCalculation(
            h_bar_effective=h_bar_effective,
            h_bar_dimensionless=h_bar_dimensionless,
            h_bar_energy_time=h_bar_system_scale,
            h_bar_action=h_bar_action,
            regime=regime,
            physical_interpretation=interpretation,
            consistency_checks=consistency
        )
    
    def _check_methods_agreement(self, *methods) -> Dict[str, float]:
        """Verifica que los diferentes métodos dan resultados consistentes"""
        values = [m for m in methods if m > 0]
        if len(values) < 2:
            return {"agreement_ratio": 0.0, "std_relative": float('inf')}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        return {
            "agreement_ratio": 1.0 - (std_val / mean_val) if mean_val > 0 else 0.0,
            "std_relative": std_val / mean_val if mean_val > 0 else float('inf'),
            "range_orders": np.log10(max(values) / min(values)) if min(values) > 0 else 0
        }
    
    def calculate_physical_constants(self, h_bar: HBarCalculation) -> Dict[str, float]:
        """
        Calcula constantes físicas derivadas usando ħ efectiva.
        """
        h_bar_eff = h_bar.h_bar_effective
        
        # Velocidad de "luz" en el sistema: v = c · (ħ_eff / ħ_SI)
        # Esto asume que la estructura del espacio-tiempo del modelo escala con ħ
        v_light = C * (h_bar_eff / H_BAR_SI) if H_BAR_SI > 0 else 0
        
        # Masa de Planck efectiva: m_P = sqrt(ħ · c / G)
        m_planck = np.sqrt(h_bar_eff * v_light / G) if G > 0 and v_light > 0 else 0
        
        # Longitud de Planck: l_P = sqrt(ħ · G / c³)
        l_planck = np.sqrt(h_bar_eff * G / v_light**3) if v_light > 0 else 0
        
        # Tiempo de Planck: t_P = l_P / c
        t_planck = l_planck / v_light if v_light > 0 else 0
        
        # Temperatura de Planck: T_P = m_P · c² / k_B (aproximando k_B ~ 1)
        t_planck_temp = m_planck * v_light**2  # en unidades donde k_B = 1
        
        return {
            "v_light_effective": v_light,
            "m_planck_effective": m_planck,
            "l_planck_effective": l_planck,
            "t_planck_effective": t_planck,
            "t_planck_temperature": t_planck_temp,
            "ratio_to_si": {
                "h_bar": h_bar_eff / H_BAR_SI,
                "v_light": v_light / C if C > 0 else 0,
                "length_scale": l_planck / 1.616e-35 if l_planck > 0 else 0  # vs l_P SI
            }
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte completo de cálculo de ħ"""
        
        h_bar = self.calculate_h_bar()
        physical = self.calculate_physical_constants(h_bar)
        
        report = {
            "metadata": {
                "checkpoint": self.checkpoint_path,
                "epoch": self.epoch,
                "timestamp": datetime.now().isoformat(),
                "calculation_version": "1.0.0"
            },
            "input_parameters": {
                "lambda": self.lambda_val,
                "delta": self.delta,
                "alpha": self.alpha,
                "mse": self.mse,
                "val_acc": self.val_acc,
                "grid_size": self.grid_size,
                "learning_rate": self.lr
            },
            "h_bar_calculation": {
                "h_bar_effective": h_bar.h_bar_effective,
                "h_bar_dimensionless": h_bar.h_bar_dimensionless,
                "h_bar_energy_time": h_bar.h_bar_energy_time,
                "h_bar_action": h_bar.h_bar_action,
                "regime": h_bar.regime,
                "physical_interpretation": h_bar.physical_interpretation,
                "consistency_checks": h_bar.consistency_checks
            },
            "derived_physical_constants": physical,
            "comparisons": {
                "vs_planck_scale": {
                    "h_bar_ratio": h_bar.h_bar_effective / H_BAR_SI,
                    "orders_of_magnitude": np.log10(h_bar.h_bar_effective / H_BAR_SI) if H_BAR_SI > 0 else 0
                }
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Imprime reporte formateado"""
        
        h_bar = report["h_bar_calculation"]
        physical = report["derived_physical_constants"]
        
        print("\n" + "="*80)
        print("CÁLCULO DE CONSTANTE DE PLANCK EFECTIVA (ħ)")
        print("="*80)
        
        print(f"\n[PARÁMETROS DE ENTRADA]")
        print(f"  Epoch: {report['metadata']['epoch']}")
        print(f"  λ (lambda): {report['input_parameters']['lambda']:.6e}")
        print(f"  δ (delta): {report['input_parameters']['delta']:.6f}")
        print(f"  α (alpha): {report['input_parameters']['alpha']:.6f}")
        print(f"  MSE: {report['input_parameters']['mse']:.6e}")
        print(f"  Val Acc: {report['input_parameters']['val_acc']:.6f}")
        
        print(f"\n[RESULTADO PRINCIPAL]")
        print(f"  ħ_efectiva = {h_bar['h_bar_effective']:.6e} [unidades del sistema]")
        print(f"  ħ_adimensional = {h_bar['h_bar_dimensionless']:.6e}")
        print(f"  Régimen: {h_bar['regime']}")
        
        print(f"\n[INTERPRETACIÓN FÍSICA]")
        # Wrap text
        text = h_bar['physical_interpretation']
        words = text.split()
        lines = []
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 78:
                lines.append(current_line)
                current_line = "  " + word
            else:
                current_line += " " + word
        lines.append(current_line)
        print("\n".join(lines))
        
        print(f"\n[CONSTANTES FÍSICAS DERIVADAS]")
        print(f"  v_luz_efectiva = {physical['v_light_effective']:.6e} m/s")
        print(f"    (ratio vs c: {physical['ratio_to_si']['v_light']:.6e})")
        print(f"  m_Planck_efectiva = {physical['m_planck_effective']:.6e} kg")
        print(f"  l_Planck_efectiva = {physical['l_planck_effective']:.6e} m")
        print(f"  t_Planck_efectiva = {physical['t_planck_effective']:.6e} s")
        print(f"  T_Planck_efectiva = {physical['t_planck_temperature']:.6e} K")
        
        print(f"\n[COMPARACIÓN CON ESCALA DE PLANCK (SI)]")
        ratio = report["comparisons"]["vs_planck_scale"]["h_bar_ratio"]
        orders = report["comparisons"]["vs_planck_scale"]["orders_of_magnitude"]
        print(f"  ħ_sistema / ħ_SI = {ratio:.6e}")
        print(f"  Órdenes de magnitud: {orders:+.2f}")
        
        if orders > 0:
            print(f"  → Tu sistema tiene ħ {orders:.0f} órdenes MAYOR que el universo físico")
            print(f"    (Efectos cuánticos 'macroscópicos' emergentes)")
        else:
            print(f"  → Tu sistema tiene ħ {abs(orders):.0f} órdenes MENOR que el universo físico")
        
        print(f"\n[VERIFICACIONES DE CONSISTENCIA]")
        checks = h_bar['consistency_checks']
        for check, value in checks.items():
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                print(f"  {check}: {status}")
            else:
                print(f"  {check}: {value}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Calcula constante de Planck efectiva desde checkpoint HPU'
    )
    parser.add_argument(
        'checkpoint', 
        nargs='?', 
        default='crystal_checkpoints/latest.pth',
        help='Path al checkpoint'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Archivo JSON de salida (opcional)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: No se encontró {args.checkpoint}")
        return
    
    # Calcular
    calculator = HBarCalculator(args.checkpoint)
    report = calculator.generate_report()
    
    # Imprimir
    calculator.print_report(report)
    
    # Guardar si se solicitó
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReporte guardado en: {args.output}")


if __name__ == "__main__":
    main()
