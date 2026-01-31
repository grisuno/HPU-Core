#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import glob
from dataclasses import dataclass
import warnings
from scipy import signal
from scipy.linalg import eig, eigvals
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from experiment2 import Config, HamiltonianNeuralNetwork, HamiltonianDataset


@dataclass
class ControlConfig:
    POLE_ZERO_TOLERANCE: float = 1e-6
    STABILITY_MARGIN: float = 0.01
    FREQUENCY_SAMPLES: int = 1000
    FREQUENCY_MIN: float = 1e-3
    FREQUENCY_MAX: float = 1e3
    TIME_SAMPLES: int = 500
    TIME_MAX: float = 10.0
    DAMPING_RATIO_CRITICAL: float = 0.707
    SETTLING_TIME_THRESHOLD: float = 0.02
    OVERSHOOT_THRESHOLD: float = 0.1
    NYQUIST_RESOLUTION: int = 500
    BODE_POINTS: int = 1000
    ROOT_LOCUS_GAIN_MIN: float = 0.0
    ROOT_LOCUS_GAIN_MAX: float = 10.0
    ROOT_LOCUS_GAIN_STEPS: int = 100
    COLORMAP: str = 'viridis'
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'


class TransferFunctionExtractor:
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_state_space_representation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        all_weights = []
        all_biases = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                all_weights.append(param.data.cpu().numpy())
            elif 'bias' in name:
                all_biases.append(param.data.cpu().numpy())
        
        if len(all_weights) == 0:
            raise ValueError("No weight matrices found in model")
        
        A_blocks = []
        for i in range(len(all_weights) - 1):
            W = all_weights[i]
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
            A_blocks.append(W)
        
        if len(A_blocks) > 0:
            A = np.vstack(A_blocks)
            n_states = A.shape[0]
            A = A[:n_states, :n_states] if A.shape[1] >= n_states else np.pad(
                A, ((0, 0), (0, n_states - A.shape[1])), mode='constant'
            )
        else:
            W = all_weights[0]
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
            A = W[:min(W.shape), :min(W.shape)]
        
        n_states = A.shape[0]
        B = np.random.randn(n_states, 1) * 0.1
        
        if len(all_weights) > 0:
            W_out = all_weights[-1]
            if W_out.ndim > 2:
                W_out = W_out.reshape(W_out.shape[0], -1)
            C = W_out[:1, :n_states] if W_out.shape[1] >= n_states else np.pad(
                W_out[:1, :], ((0, 0), (0, n_states - W_out.shape[1])), mode='constant'
            )
        else:
            C = np.random.randn(1, n_states) * 0.1
        
        D = np.zeros((1, 1))
        
        return A, B, C, D
    
    def compute_transfer_function(
        self, 
        A: np.ndarray, 
        B: np.ndarray, 
        C: np.ndarray, 
        D: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        n = A.shape[0]
        s = np.poly([0])
        I = np.eye(n)
        
        try:
            sI_minus_A = np.zeros((n, n), dtype=complex)
            for i in range(n):
                for j in range(n):
                    sI_minus_A[i, j] = -A[i, j]
                sI_minus_A[i, i] += 1.0
            
            eigenvalues = eigvals(A)
            
            num_coeffs = np.poly(eigenvalues)
            
            char_poly = np.poly(A)
            den_coeffs = char_poly
            
            num_coeffs = np.real(num_coeffs)
            den_coeffs = np.real(den_coeffs)
            
        except Exception as e:
            warnings.warn(f"Transfer function computation failed: {e}")
            num_coeffs = np.array([1.0])
            den_coeffs = np.array([1.0, 1.0])
        
        return num_coeffs, den_coeffs


class PoleZeroAnalyzer:
    
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray):
        self.num = numerator
        self.den = denominator
        self.poles = None
        self.zeros = None
        self.system = None
        
        self._compute_poles_zeros()
    
    def _compute_poles_zeros(self):
        
        self.zeros = np.roots(self.num) if len(self.num) > 1 else np.array([])
        self.poles = np.roots(self.den) if len(self.den) > 1 else np.array([])
        
        self.zeros = self.zeros[np.abs(self.zeros) < 1e10]
        self.poles = self.poles[np.abs(self.poles) < 1e10]
        
        try:
            self.system = signal.TransferFunction(self.num, self.den)
        except:
            self.system = None
    
    def analyze_stability(self) -> Dict[str, Any]:
        
        if len(self.poles) == 0:
            return {
                'is_stable': True,
                'stability_type': 'trivially_stable',
                'dominant_pole': None,
                'stability_margin': float('inf'),
                'unstable_poles': []
            }
        
        real_parts = np.real(self.poles)
        
        is_stable = np.all(real_parts < -ControlConfig.STABILITY_MARGIN)
        
        unstable_poles = self.poles[real_parts >= -ControlConfig.STABILITY_MARGIN]
        
        if is_stable:
            stability_type = 'asymptotically_stable'
        elif np.any(real_parts > ControlConfig.STABILITY_MARGIN):
            stability_type = 'unstable'
        else:
            stability_type = 'marginally_stable'
        
        if len(real_parts) > 0:
            dominant_idx = np.argmax(real_parts)
            dominant_pole = self.poles[dominant_idx]
            stability_margin = -real_parts[dominant_idx]
        else:
            dominant_pole = None
            stability_margin = float('inf')
        
        return {
            'is_stable': bool(is_stable),
            'stability_type': stability_type,
            'dominant_pole': complex(dominant_pole) if dominant_pole is not None else None,
            'stability_margin': float(stability_margin),
            'unstable_poles': [complex(p) for p in unstable_poles],
            'num_unstable': len(unstable_poles)
        }
    
    def classify_poles(self) -> Dict[str, List[complex]]:
        
        real_poles = []
        complex_poles = []
        
        processed = set()
        
        for i, pole in enumerate(self.poles):
            if i in processed:
                continue
            
            if np.abs(np.imag(pole)) < ControlConfig.POLE_ZERO_TOLERANCE:
                real_poles.append(complex(np.real(pole), 0))
                processed.add(i)
            else:
                complex_poles.append(complex(pole))
                processed.add(i)
                
                for j in range(i + 1, len(self.poles)):
                    if j not in processed:
                        if np.abs(pole - np.conj(self.poles[j])) < ControlConfig.POLE_ZERO_TOLERANCE:
                            processed.add(j)
                            break
        
        return {
            'real_poles': real_poles,
            'complex_conjugate_pairs': complex_poles,
            'total_real': len(real_poles),
            'total_complex': len(complex_poles)
        }
    
    def compute_damping_frequency(self) -> List[Dict[str, float]]:
        
        results = []
        
        for pole in self.poles:
            real_part = np.real(pole)
            imag_part = np.imag(pole)
            
            if np.abs(imag_part) < ControlConfig.POLE_ZERO_TOLERANCE:
                results.append({
                    'pole': complex(pole),
                    'natural_frequency': abs(real_part),
                    'damping_ratio': 1.0 if real_part < 0 else -1.0,
                    'damped_frequency': 0.0,
                    'type': 'overdamped'
                })
            else:
                omega_n = np.sqrt(real_part**2 + imag_part**2)
                zeta = -real_part / omega_n if omega_n > 0 else 0
                omega_d = abs(imag_part)
                
                if zeta > 1:
                    pole_type = 'overdamped'
                elif abs(zeta - 1.0) < ControlConfig.POLE_ZERO_TOLERANCE:
                    pole_type = 'critically_damped'
                elif 0 < zeta < 1:
                    pole_type = 'underdamped'
                else:
                    pole_type = 'undamped'
                
                results.append({
                    'pole': complex(pole),
                    'natural_frequency': float(omega_n),
                    'damping_ratio': float(zeta),
                    'damped_frequency': float(omega_d),
                    'type': pole_type
                })
        
        return results
    
    def compute_time_constants(self) -> List[Dict[str, float]]:
        
        time_constants = []
        
        for pole in self.poles:
            real_part = np.real(pole)
            
            if real_part < -ControlConfig.POLE_ZERO_TOLERANCE:
                tau = -1.0 / real_part
                settling_time = 4 * tau
                
                time_constants.append({
                    'pole': complex(pole),
                    'time_constant': float(tau),
                    'settling_time_4tau': float(settling_time),
                    'bandwidth': float(1.0 / tau)
                })
        
        return time_constants


class FrequencyResponseAnalyzer:
    
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray):
        self.num = numerator
        self.den = denominator
        try:
            self.system = signal.TransferFunction(numerator, denominator)
        except:
            self.system = None
    
    def compute_bode_plot_data(self) -> Dict[str, np.ndarray]:
        
        if self.system is None:
            omega = np.logspace(
                np.log10(ControlConfig.FREQUENCY_MIN),
                np.log10(ControlConfig.FREQUENCY_MAX),
                ControlConfig.BODE_POINTS
            )
            magnitude = np.zeros_like(omega)
            phase = np.zeros_like(omega)
        else:
            omega, magnitude, phase = signal.bode(self.system)
        
        return {
            'frequency': omega,
            'magnitude_db': magnitude,
            'phase_deg': phase
        }
    
    def compute_gain_phase_margins(self) -> Dict[str, float]:
        
        if self.system is None:
            return {
                'gain_margin_db': float('inf'),
                'phase_margin_deg': float('inf'),
                'gain_crossover_freq': 0.0,
                'phase_crossover_freq': 0.0
            }
        
        try:
            gm, pm, wgc, wpc = signal.margin(self.system)
            
            gm_db = 20 * np.log10(gm) if gm > 0 else float('inf')
            
            return {
                'gain_margin_db': float(gm_db),
                'phase_margin_deg': float(pm),
                'gain_crossover_freq': float(wgc),
                'phase_crossover_freq': float(wpc)
            }
        except:
            return {
                'gain_margin_db': float('inf'),
                'phase_margin_deg': float('inf'),
                'gain_crossover_freq': 0.0,
                'phase_crossover_freq': 0.0
            }
    
    def compute_nyquist_data(self) -> Dict[str, np.ndarray]:
        
        omega = np.logspace(
            np.log10(ControlConfig.FREQUENCY_MIN),
            np.log10(ControlConfig.FREQUENCY_MAX),
            ControlConfig.NYQUIST_RESOLUTION
        )
        
        if self.system is None:
            real = np.zeros_like(omega)
            imag = np.zeros_like(omega)
        else:
            _, H = signal.freqresp(self.system, omega)
            real = np.real(H)
            imag = np.imag(H)
        
        return {
            'frequency': omega,
            'real': real,
            'imag': imag
        }
    
    def evaluate_nyquist_stability(self, nyquist_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        
        real = nyquist_data['real']
        imag = nyquist_data['imag']
        
        encirclements = 0
        critical_point = -1.0 + 0j
        
        for i in range(len(real) - 1):
            z1 = complex(real[i], imag[i]) - critical_point
            z2 = complex(real[i + 1], imag[i + 1]) - critical_point
            
            angle1 = np.angle(z1)
            angle2 = np.angle(z2)
            
            delta_angle = angle2 - angle1
            
            if delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            
            encirclements += delta_angle
        
        encirclements = int(np.round(encirclements / (2 * np.pi)))
        
        min_distance = np.min(np.sqrt((real + 1)**2 + imag**2))
        
        return {
            'encirclements': encirclements,
            'is_stable_nyquist': encirclements == 0,
            'distance_to_critical': float(min_distance),
            'stability_robustness': float(min_distance)
        }


class TimeResponseAnalyzer:
    
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray):
        self.num = numerator
        self.den = denominator
        try:
            self.system = signal.TransferFunction(numerator, denominator)
        except:
            self.system = None
    
    def compute_step_response(self) -> Dict[str, np.ndarray]:
        
        t = np.linspace(0, ControlConfig.TIME_MAX, ControlConfig.TIME_SAMPLES)
        
        if self.system is None:
            y = np.zeros_like(t)
        else:
            t_out, y_out = signal.step(self.system, T=t)
            t = t_out
            y = y_out
        
        return {
            'time': t,
            'output': y
        }
    
    def compute_impulse_response(self) -> Dict[str, np.ndarray]:
        
        t = np.linspace(0, ControlConfig.TIME_MAX, ControlConfig.TIME_SAMPLES)
        
        if self.system is None:
            y = np.zeros_like(t)
        else:
            t_out, y_out = signal.impulse(self.system, T=t)
            t = t_out
            y = y_out
        
        return {
            'time': t,
            'output': y
        }
    
    def analyze_step_response_characteristics(
        self, 
        step_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        
        t = step_data['time']
        y = step_data['output']
        
        if len(y) == 0 or np.all(y == 0):
            return {
                'rise_time': 0.0,
                'settling_time': 0.0,
                'overshoot_percent': 0.0,
                'peak_time': 0.0,
                'steady_state_value': 0.0,
                'steady_state_error': 0.0
            }
        
        steady_state = y[-1]
        
        if abs(steady_state) < 1e-10:
            steady_state = 1.0
        
        threshold_10 = 0.1 * steady_state
        threshold_90 = 0.9 * steady_state
        
        rise_time = 0.0
        idx_10 = np.where(y >= threshold_10)[0]
        idx_90 = np.where(y >= threshold_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = t[idx_90[0]] - t[idx_10[0]]
        
        settling_band = ControlConfig.SETTLING_TIME_THRESHOLD * abs(steady_state)
        settling_time = t[-1]
        
        for i in range(len(y) - 1, -1, -1):
            if abs(y[i] - steady_state) > settling_band:
                settling_time = t[i]
                break
        
        peak_value = np.max(y)
        peak_idx = np.argmax(y)
        peak_time = t[peak_idx]
        
        overshoot = ((peak_value - steady_state) / abs(steady_state)) * 100 if steady_state != 0 else 0
        
        steady_state_error = abs(1.0 - steady_state)
        
        return {
            'rise_time': float(rise_time),
            'settling_time': float(settling_time),
            'overshoot_percent': float(overshoot),
            'peak_time': float(peak_time),
            'steady_state_value': float(steady_state),
            'steady_state_error': float(steady_state_error)
        }


class ControllerDesigner:
    
    def __init__(self, poles: np.ndarray, zeros: np.ndarray):
        self.poles = poles
        self.zeros = zeros
    
    def design_pid_controller(
        self, 
        desired_damping: float = 0.707,
        desired_settling_time: float = 2.0
    ) -> Dict[str, float]:
        
        omega_n = 4.0 / (desired_damping * desired_settling_time)
        
        kp = omega_n**2
        ki = omega_n**3 / 10.0
        kd = 2 * desired_damping * omega_n
        
        return {
            'kp': float(kp),
            'ki': float(ki),
            'kd': float(kd),
            'omega_n': float(omega_n),
            'damping': float(desired_damping)
        }
    
    def design_lead_compensator(
        self, 
        desired_phase_margin: float = 45.0
    ) -> Dict[str, float]:
        
        phi_max = np.deg2rad(desired_phase_margin)
        
        alpha = (1 - np.sin(phi_max)) / (1 + np.sin(phi_max))
        
        if len(self.poles) > 0:
            dominant_pole = self.poles[np.argmax(np.real(self.poles))]
            omega_m = abs(dominant_pole)
        else:
            omega_m = 1.0
        
        T = 1.0 / (omega_m * np.sqrt(alpha))
        
        zero = -1.0 / T
        pole = -1.0 / (alpha * T)
        
        kc = 1.0 / alpha
        
        return {
            'gain': float(kc),
            'zero': float(zero),
            'pole': float(pole),
            'alpha': float(alpha),
            'omega_m': float(omega_m)
        }
    
    def compute_root_locus(
        self, 
        num: np.ndarray, 
        den: np.ndarray
    ) -> Dict[str, np.ndarray]:
        
        gains = np.linspace(
            ControlConfig.ROOT_LOCUS_GAIN_MIN,
            ControlConfig.ROOT_LOCUS_GAIN_MAX,
            ControlConfig.ROOT_LOCUS_GAIN_STEPS
        )
        
        root_locus_poles = []
        
        for k in gains:
            num_cl = num
            den_cl = den + k * num
            
            poles_cl = np.roots(den_cl) if len(den_cl) > 1 else np.array([])
            root_locus_poles.append(poles_cl)
        
        return {
            'gains': gains,
            'poles': root_locus_poles
        }


class ControlSystemAnalyzer:
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        self.model = HamiltonianNeuralNetwork(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(self.device)
        
        if 'model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.metrics = self.checkpoint.get('metrics', {})
    
    def analyze_complete_system(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        print(f"\nAnalyzing checkpoint: {self.checkpoint_path}")
        print(f"Epoch: {self.epoch}\n")
        
        extractor = TransferFunctionExtractor(self.model, self.device)
        A, B, C, D = extractor.extract_state_space_representation()
        
        num, den = extractor.compute_transfer_function(A, B, C, D)
        
        pz_analyzer = PoleZeroAnalyzer(num, den)
        
        stability = pz_analyzer.analyze_stability()
        pole_classification = pz_analyzer.classify_poles()
        damping_freq = pz_analyzer.compute_damping_frequency()
        time_constants = pz_analyzer.compute_time_constants()
        
        freq_analyzer = FrequencyResponseAnalyzer(num, den)
        bode_data = freq_analyzer.compute_bode_plot_data()
        margins = freq_analyzer.compute_gain_phase_margins()
        nyquist_data = freq_analyzer.compute_nyquist_data()
        nyquist_stability = freq_analyzer.evaluate_nyquist_stability(nyquist_data)
        
        time_analyzer = TimeResponseAnalyzer(num, den)
        step_response = time_analyzer.compute_step_response()
        impulse_response = time_analyzer.compute_impulse_response()
        step_characteristics = time_analyzer.analyze_step_response_characteristics(step_response)
        
        controller = ControllerDesigner(pz_analyzer.poles, pz_analyzer.zeros)
        pid_params = controller.design_pid_controller()
        lead_comp = controller.design_lead_compensator()
        root_locus = controller.compute_root_locus(num, den)
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '1.0.0'
            },
            'state_space': {
                'A_shape': A.shape,
                'B_shape': B.shape,
                'C_shape': C.shape,
                'D_shape': D.shape,
                'num_states': A.shape[0]
            },
            'transfer_function': {
                'numerator': num.tolist(),
                'denominator': den.tolist(),
                'order': len(den) - 1
            },
            'poles_zeros': {
                'poles': [complex(p) for p in pz_analyzer.poles],
                'zeros': [complex(z) for z in pz_analyzer.zeros],
                'num_poles': len(pz_analyzer.poles),
                'num_zeros': len(pz_analyzer.zeros)
            },
            'stability': stability,
            'pole_classification': {
                'real_poles': [complex(p) for p in pole_classification['real_poles']],
                'complex_poles': [complex(p) for p in pole_classification['complex_conjugate_pairs']],
                'total_real': pole_classification['total_real'],
                'total_complex': pole_classification['total_complex']
            },
            'damping_frequency': damping_freq,
            'time_constants': time_constants,
            'frequency_margins': margins,
            'nyquist_stability': nyquist_stability,
            'step_response_characteristics': step_characteristics,
            'controller_design': {
                'pid': pid_params,
                'lead_compensator': lead_comp
            }
        }
        
        plot_data = {
            'poles': pz_analyzer.poles,
            'zeros': pz_analyzer.zeros,
            'bode': bode_data,
            'nyquist': nyquist_data,
            'step_response': step_response,
            'impulse_response': impulse_response,
            'root_locus': root_locus
        }
        
        self._print_report(results)
        
        return results, plot_data
    
    def _print_report(self, results: Dict):
        
        print("=" * 70)
        print("CONTROL SYSTEM ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[STATE SPACE REPRESENTATION]")
        ss = results['state_space']
        print(f"  A matrix: {ss['A_shape']}")
        print(f"  B matrix: {ss['B_shape']}")
        print(f"  C matrix: {ss['C_shape']}")
        print(f"  D matrix: {ss['D_shape']}")
        print(f"  Number of states: {ss['num_states']}")
        
        print(f"\n[TRANSFER FUNCTION]")
        tf = results['transfer_function']
        print(f"  Order: {tf['order']}")
        print(f"  Numerator coefficients: {len(tf['numerator'])}")
        print(f"  Denominator coefficients: {len(tf['denominator'])}")
        
        print(f"\n[POLES AND ZEROS]")
        pz = results['poles_zeros']
        print(f"  Number of poles: {pz['num_poles']}")
        print(f"  Number of zeros: {pz['num_zeros']}")
        
        print(f"\n[STABILITY ANALYSIS]")
        stab = results['stability']
        print(f"  Stable: {stab['is_stable']}")
        print(f"  Stability type: {stab['stability_type']}")
        print(f"  Stability margin: {stab['stability_margin']:.6f}")
        print(f"  Unstable poles: {stab['num_unstable']}")
        if stab['dominant_pole'] is not None:
            dp = stab['dominant_pole']
            print(f"  Dominant pole: {dp.real:.6f} + {dp.imag:.6f}j")
        
        print(f"\n[POLE CLASSIFICATION]")
        pc = results['pole_classification']
        print(f"  Real poles: {pc['total_real']}")
        print(f"  Complex conjugate pairs: {pc['total_complex']}")
        
        print(f"\n[FREQUENCY DOMAIN]")
        margins = results['frequency_margins']
        print(f"  Gain margin: {margins['gain_margin_db']:.2f} dB")
        print(f"  Phase margin: {margins['phase_margin_deg']:.2f} deg")
        print(f"  Gain crossover: {margins['gain_crossover_freq']:.6f} rad/s")
        print(f"  Phase crossover: {margins['phase_crossover_freq']:.6f} rad/s")
        
        print(f"\n[NYQUIST STABILITY]")
        nyq = results['nyquist_stability']
        print(f"  Encirclements: {nyq['encirclements']}")
        print(f"  Stable: {nyq['is_stable_nyquist']}")
        print(f"  Distance to critical point: {nyq['distance_to_critical']:.6f}")
        print(f"  Robustness: {nyq['stability_robustness']:.6f}")
        
        print(f"\n[STEP RESPONSE]")
        step = results['step_response_characteristics']
        print(f"  Rise time: {step['rise_time']:.6f} s")
        print(f"  Settling time: {step['settling_time']:.6f} s")
        print(f"  Overshoot: {step['overshoot_percent']:.2f} %")
        print(f"  Peak time: {step['peak_time']:.6f} s")
        print(f"  Steady state value: {step['steady_state_value']:.6f}")
        print(f"  Steady state error: {step['steady_state_error']:.6f}")
        
        print(f"\n[CONTROLLER DESIGN]")
        pid = results['controller_design']['pid']
        print(f"  PID Controller:")
        print(f"    Kp: {pid['kp']:.6f}")
        print(f"    Ki: {pid['ki']:.6f}")
        print(f"    Kd: {pid['kd']:.6f}")
        
        lead = results['controller_design']['lead_compensator']
        print(f"  Lead Compensator:")
        print(f"    Gain: {lead['gain']:.6f}")
        print(f"    Zero: {lead['zero']:.6f}")
        print(f"    Pole: {lead['pole']:.6f}")
        
        print("=" * 70)


class ControlVisualizer:
    
    @staticmethod
    def plot_pole_zero_map(
        poles: np.ndarray,
        zeros: np.ndarray,
        output_path: str
    ):
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=ControlConfig.FIGURE_DPI)
        
        if len(poles) > 0:
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                s=150,
                marker='x',
                color='#D62828',
                linewidths=2.5,
                label='Poles',
                zorder=5
            )
        
        if len(zeros) > 0:
            ax.scatter(
                np.real(zeros),
                np.imag(zeros),
                s=150,
                marker='o',
                facecolors='none',
                edgecolors='#06A77D',
                linewidths=2.5,
                label='Zeros',
                zorder=5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        unit_circle = Circle(
            (0, 0), 
            1, 
            fill=False, 
            edgecolor='black', 
            linestyle=':', 
            linewidth=1.5,
            alpha=0.3
        )
        ax.add_patch(unit_circle)
        
        ax.axvline(x=-ControlConfig.STABILITY_MARGIN, color='#F18F01', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Stability margin')
        
        ax.set_xlabel('Real axis', fontsize=12)
        ax.set_ylabel('Imaginary axis', fontsize=12)
        ax.set_title('Pole-Zero Map', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved pole-zero map: {output_path}")
    
    @staticmethod
    def plot_bode_diagram(
        bode_data: Dict[str, np.ndarray],
        margins: Dict[str, float],
        output_path: str
    ):
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 10),
            dpi=ControlConfig.FIGURE_DPI
        )
        
        ax1.semilogx(
            bode_data['frequency'],
            bode_data['magnitude_db'],
            color='#2E86AB',
            linewidth=2
        )
        ax1.axhline(
            y=0,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.5
        )
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title('Bode Diagram', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3, linestyle=':')
        
        if margins['gain_crossover_freq'] > 0:
            ax1.axvline(
                x=margins['gain_crossover_freq'],
                color='#F18F01',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f"Gain crossover: {margins['gain_crossover_freq']:.2f} rad/s"
            )
        ax1.legend(loc='best', fontsize=9)
        
        ax2.semilogx(
            bode_data['frequency'],
            bode_data['phase_deg'],
            color='#A23B72',
            linewidth=2
        )
        ax2.axhline(
            y=-180,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.5
        )
        ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
        ax2.set_ylabel('Phase (deg)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3, linestyle=':')
        
        if margins['phase_crossover_freq'] > 0:
            ax2.axvline(
                x=margins['phase_crossover_freq'],
                color='#F18F01',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f"Phase crossover: {margins['phase_crossover_freq']:.2f} rad/s"
            )
        ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved Bode diagram: {output_path}")
    
    @staticmethod
    def plot_nyquist_diagram(
        nyquist_data: Dict[str, np.ndarray],
        output_path: str
    ):
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=ControlConfig.FIGURE_DPI)
        
        real = nyquist_data['real']
        imag = nyquist_data['imag']
        
        ax.plot(real, imag, color='#2E86AB', linewidth=2, label='Nyquist plot')
        ax.plot(real, -imag, color='#2E86AB', linewidth=2, linestyle='--', alpha=0.5)
        
        ax.plot(
            -1, 0,
            marker='x',
            markersize=15,
            color='#D62828',
            markeredgewidth=3,
            label='Critical point (-1, 0)'
        )
        
        critical_circle = Circle(
            (-1, 0),
            0.5,
            fill=False,
            edgecolor='#F18F01',
            linestyle=':',
            linewidth=2,
            alpha=0.5
        )
        ax.add_patch(critical_circle)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Real axis', fontsize=12)
        ax.set_ylabel('Imaginary axis', fontsize=12)
        ax.set_title('Nyquist Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved Nyquist diagram: {output_path}")
    
    @staticmethod
    def plot_time_responses(
        step_data: Dict[str, np.ndarray],
        impulse_data: Dict[str, np.ndarray],
        output_path: str
    ):
        
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(12, 10),
            dpi=ControlConfig.FIGURE_DPI
        )
        
        ax1.plot(
            step_data['time'],
            step_data['output'],
            color='#06A77D',
            linewidth=2,
            label='Step response'
        )
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target')
        ax1.set_ylabel('Output', fontsize=12)
        ax1.set_title('Step Response', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        ax2.plot(
            impulse_data['time'],
            impulse_data['output'],
            color='#A23B72',
            linewidth=2,
            label='Impulse response'
        )
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Output', fontsize=12)
        ax2.set_title('Impulse Response', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved time responses: {output_path}")
    
    @staticmethod
    def plot_root_locus(
        root_locus_data: Dict[str, np.ndarray],
        output_path: str
    ):
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=ControlConfig.FIGURE_DPI)
        
        gains = root_locus_data['gains']
        poles_list = root_locus_data['poles']
        
        for i in range(len(poles_list[0]) if len(poles_list) > 0 else 0):
            branch_real = []
            branch_imag = []
            
            for poles in poles_list:
                if i < len(poles):
                    branch_real.append(np.real(poles[i]))
                    branch_imag.append(np.imag(poles[i]))
            
            if len(branch_real) > 0:
                ax.plot(
                    branch_real,
                    branch_imag,
                    color='#2E86AB',
                    linewidth=1.5,
                    alpha=0.7
                )
        
        if len(poles_list) > 0 and len(poles_list[0]) > 0:
            start_poles = poles_list[0]
            ax.scatter(
                np.real(start_poles),
                np.imag(start_poles),
                s=100,
                marker='x',
                color='#D62828',
                linewidths=2,
                label='Starting poles',
                zorder=5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-ControlConfig.STABILITY_MARGIN, color='#F18F01', 
                   linestyle='--', linewidth=2, alpha=0.7, label='Stability margin')
        
        ax.set_xlabel('Real axis', fontsize=12)
        ax.set_ylabel('Imaginary axis', fontsize=12)
        ax.set_title('Root Locus', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved root locus: {output_path}")
    
    @staticmethod
    def plot_combined_analysis(
        poles: np.ndarray,
        zeros: np.ndarray,
        bode_data: Dict[str, np.ndarray],
        step_data: Dict[str, np.ndarray],
        output_path: str
    ):
        
        fig = plt.figure(figsize=(16, 12), dpi=ControlConfig.FIGURE_DPI)
        
        ax1 = plt.subplot(2, 2, 1)
        if len(poles) > 0:
            ax1.scatter(np.real(poles), np.imag(poles), s=100, marker='x', 
                       color='#D62828', linewidths=2, label='Poles')
        if len(zeros) > 0:
            ax1.scatter(np.real(zeros), np.imag(zeros), s=100, marker='o',
                       facecolors='none', edgecolors='#06A77D', linewidths=2, label='Zeros')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Real', fontsize=10)
        ax1.set_ylabel('Imaginary', fontsize=10)
        ax1.set_title('Pole-Zero Map', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 2, 2)
        ax2.semilogx(bode_data['frequency'], bode_data['magnitude_db'], 
                    color='#2E86AB', linewidth=1.5)
        ax2.set_ylabel('Magnitude (dB)', fontsize=10)
        ax2.set_title('Bode Magnitude', fontsize=11, fontweight='bold')
        ax2.grid(True, which='both', alpha=0.3)
        
        ax3 = plt.subplot(2, 2, 3)
        ax3.semilogx(bode_data['frequency'], bode_data['phase_deg'],
                    color='#A23B72', linewidth=1.5)
        ax3.set_xlabel('Frequency (rad/s)', fontsize=10)
        ax3.set_ylabel('Phase (deg)', fontsize=10)
        ax3.set_title('Bode Phase', fontsize=11, fontweight='bold')
        ax3.grid(True, which='both', alpha=0.3)
        
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(step_data['time'], step_data['output'], 
                color='#06A77D', linewidth=2)
        ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Output', fontsize=10)
        ax4.set_title('Step Response', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=ControlConfig.FIGURE_DPI, format=ControlConfig.SAVE_FORMAT)
        plt.close()
        
        print(f"Saved combined analysis: {output_path}")


def analyze_checkpoint(
    checkpoint_path: str,
    output_dir: str = 'control_analysis'
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ControlSystemAnalyzer(checkpoint_path)
    results, plot_data = analyzer.analyze_complete_system()
    
    base_name = Path(checkpoint_path).stem
    
    ControlVisualizer.plot_pole_zero_map(
        plot_data['poles'],
        plot_data['zeros'],
        os.path.join(output_dir, f'{base_name}_pole_zero.{ControlConfig.SAVE_FORMAT}')
    )
    
    ControlVisualizer.plot_bode_diagram(
        plot_data['bode'],
        results['frequency_margins'],
        os.path.join(output_dir, f'{base_name}_bode.{ControlConfig.SAVE_FORMAT}')
    )
    
    ControlVisualizer.plot_nyquist_diagram(
        plot_data['nyquist'],
        os.path.join(output_dir, f'{base_name}_nyquist.{ControlConfig.SAVE_FORMAT}')
    )
    
    ControlVisualizer.plot_time_responses(
        plot_data['step_response'],
        plot_data['impulse_response'],
        os.path.join(output_dir, f'{base_name}_time_response.{ControlConfig.SAVE_FORMAT}')
    )
    
    ControlVisualizer.plot_root_locus(
        plot_data['root_locus'],
        os.path.join(output_dir, f'{base_name}_root_locus.{ControlConfig.SAVE_FORMAT}')
    )
    
    ControlVisualizer.plot_combined_analysis(
        plot_data['poles'],
        plot_data['zeros'],
        plot_data['bode'],
        plot_data['step_response'],
        os.path.join(output_dir, f'{base_name}_combined.{ControlConfig.SAVE_FORMAT}')
    )
    
    results_path = os.path.join(output_dir, f'{base_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results: {results_path}")
    
    return results


def analyze_multiple_checkpoints(
    checkpoint_dir: str = 'crystal_checkpoints',
    n_latest: int = 5,
    output_dir: str = 'control_analysis'
):
    
    pattern = os.path.join(checkpoint_dir, '*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    checkpoints = checkpoints[:n_latest]
    
    print(f"\nAnalyzing {len(checkpoints)} checkpoints...\n")
    
    all_results = []
    for cp in checkpoints:
        try:
            results = analyze_checkpoint(cp, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {cp}: {e}")
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Control theory analysis for HPU checkpoints'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Analyze N latest checkpoints'
    )
    parser.add_argument(
        '--dir',
        default='crystal_checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--output',
        default='control_analysis',
        help='Output directory for plots and results'
    )
    
    args = parser.parse_args()
    
    if args.latest:
        analyze_multiple_checkpoints(args.dir, args.latest, args.output)
    elif args.checkpoint:
        analyze_checkpoint(args.checkpoint, args.output)
    else:
        latest = os.path.join(args.dir, 'latest.pth')
        if os.path.exists(latest):
            analyze_checkpoint(latest, args.output)
        else:
            analyze_multiple_checkpoints(args.dir, 1, args.output)


if __name__ == '__main__':
    main()