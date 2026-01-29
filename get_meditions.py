import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol
from pathlib import Path
import glob
from dataclasses import dataclass
from collections import deque
import warnings
import logging

from scipy.stats import gaussian_kde
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA

from experiment2 import Config, HamiltonianNeuralNetwork, HamiltonianDataset

# ============================================================================
# CONFIGURACIÓN Y UTILIDADES (del sistema Strassen adaptadas)
# ============================================================================

@dataclass
class ThermodynamicConfig:
    """Configuración termodinámica para análisis de HPU Core"""
    HBAR: float = 1e-6
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    PCA_COMPONENTS: int = 2
    ENTROPY_BINS: int = 50
    ENTROPY_EPS: float = 1e-10
    KDE_BANDWIDTH: str = 'scott'
    ENTROPY_METHOD: str = 'shannon'
    
    # Umbrales de fase
    CRYSTAL_ALPHA_THRESHOLD: float = 7.0
    GLASS_ALPHA_THRESHOLD: float = 3.0
    CV_CRITICAL_THRESHOLD: float = 1.0
    
    # Parámetros físicos para ecuación de estado
    T_0: float = 1e-3  # Temperatura de referencia
    C_COEFF: float = 0.5  # Coeficiente de acoplamiento α-T
    


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger(__name__)

# ============================================================================
# CLASES DE DATOS TERMODINÁMICOS
# ============================================================================

@dataclass
class ThermodynamicPotential:
    """Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C"""
    internal_energy: float  # Loss de generalización
    temperature: float      # T_eff
    entropy: float          # S_gen
    chemical_potential: float  # μ para "átomos" de conocimiento
    crystallinity: float    # α (pureza cristalina)
    particle_number: float  # N (parámetros activos)
    
    def helmholtz_free_energy(self) -> float:
        """F = U - T*S (a μ y N constantes)"""
        return self.internal_energy - self.temperature * self.entropy
    
    def gibbs_free_energy(self) -> float:
        """G = F + μ*N + P*V (presión algorítmica)"""
        pressure_term = self.crystallinity * self.particle_number
        return self.helmholtz_free_energy() + self.chemical_potential * self.particle_number + pressure_term
    
    def is_stable(self) -> bool:
        """Criterio de estabilidad: dG < 0"""
        return self.gibbs_free_energy() < 0

# ============================================================================
# MÉTRICAS CRISTALOGRÁFICAS (adaptadas de Strassen)
# ============================================================================

class CrystallographyMetrics:
    """
    Métricas de cristalografía para redes neuronales Hamiltonianas.
    Mide la "pureza" estructural de los pesos aprendidos.
    """
    
    @dataclass
    class SpectralCoefficients:
        """Contenedor para coeficientes espectrales de HPU Core"""
        spectral_weights: torch.Tensor  # Pesos de capas espectrales
        encoder_weights: torch.Tensor   # Pesos del encoder
        decoder_weights: torch.Tensor   # Pesos del decoder
        
        @classmethod
        def from_model(cls, model: HamiltonianNeuralNetwork) -> 'CrystallographyMetrics.SpectralCoefficients':
            """Extrae coeficientes del modelo HPU Core"""
            # Extraer pesos de capas espectrales
            spectral_w = []
            for layer in model.spectral_layers:
                spectral_w.append(layer.weight.data.flatten())
            spectral_weights = torch.cat(spectral_w) if spectral_w else torch.tensor([])
            
            # Encoder y decoder
            encoder_w = model.encoder.weight.data.flatten()
            decoder_w = model.decoder.weight.data.flatten()
            
            return cls(spectral_weights, encoder_w, decoder_w)
    
    @staticmethod
    def compute_kappa(model: HamiltonianNeuralNetwork, val_x: torch.Tensor, 
                     val_y: torch.Tensor, num_batches: int = 5) -> float:
        """
        Número de condición de la matriz de covarianza de gradientes.
        FIX: Limita tamaño de gradientes y usa método iterativo si es necesario.
        """
        model.eval()
        grads = []
        
        # Generar múltiples gradientes con pequeñas perturbaciones
        for _ in range(num_batches):
            model.zero_grad()
            outputs = model(val_x)
            loss = F.mse_loss(outputs, val_y)
            loss.backward(retain_graph=True)
            
            grad = torch.cat([p.grad.flatten() for p in model.parameters() 
                            if p.grad is not None])
            grads.append(grad.detach())
        
        if len(grads) < 2:
            return float('inf')
        
        grads_tensor = torch.stack(grads)
        n_samples, n_dims = grads_tensor.shape
        
        # Si hay muchos parámetros, submuestreamos o usamos método iterativo
        if n_dims > Config.KAPPA_MAX_DIM:
            # Submuestrear dimensiones aleatoriamente
            indices = torch.randperm(n_dims)[:Config.KAPPA_MAX_DIM]
            grads_tensor = grads_tensor[:, indices]
            n_dims = Config.KAPPA_MAX_DIM
        
        # Matriz de covarianza: usar forma eficiente
        # Cov = (G^T G) / (n_samples - 1), donde G es grads_tensor
        # Para n_samples << n_dims, calculamos eigenvalores de G G^T en lugar de G^T G
        try:
            if n_samples < n_dims:
                # Forma económica: eigenvalores de G G^T son los mismos que G^T G
                # más (n_dims - n_samples) ceros
                cov_small = torch.mm(grads_tensor, grads_tensor.t()) / (n_samples - 1)
                eigenvalues_small = torch.linalg.eigvalsh(cov_small)
                # Los eigenvalores no nulos son los mismos
                eigenvalues = eigenvalues_small[eigenvalues_small > Config.EIGENVALUE_TOL]
            else:
                cov_matrix = torch.cov(grads_tensor.T)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix).real
            
            eigenvalues = eigenvalues[eigenvalues > Config.EIGENVALUE_TOL]
            
            if len(eigenvalues) == 0:
                return float('inf')
            
            kappa = eigenvalues.max() / eigenvalues.min()
            return kappa.item()
        except Exception as e:
            logger.warning(f"Error computing kappa: {e}")
            return float('inf')

    @staticmethod
    def compute_discretization_margin(model: HamiltonianNeuralNetwork) -> float:
        """
        δ = max |w - round(w)| sobre todos los parámetros.
        Mide qué tan cerca están los pesos de valores enteros.
        """
        all_deltas = []
        
        for param in model.parameters():
            if param.numel() > 0:
                rounded = torch.round(param.data)
                delta = (param.data - rounded).abs().max().item()
                all_deltas.append(delta)
        
        return max(all_deltas) if all_deltas else 1.0
    
    @staticmethod
    def compute_alpha_purity(model: HamiltonianNeuralNetwork) -> float:
        """
        α = -log(δ). Pureza cristalina.
        α > 7 indica estructura cristalina perfecta.
        """
        delta = CrystallographyMetrics.compute_discretization_margin(model)
        if delta < 1e-10:
            return 20.0
        return -np.log(delta + 1e-15)
    
    @staticmethod
    def compute_local_complexity(model: HamiltonianNeuralNetwork) -> float:
        """
        Fracción de parámetros "activos" (no cerca de cero).
        """
        params = torch.cat([p.flatten() for p in model.parameters()])
        with torch.no_grad():
            perc_95 = torch.quantile(torch.abs(params), 0.95)
            active = (torch.abs(params) > 0.01 * perc_95).sum()
            lc = active.float() / len(params)
        return lc.item()
    
    @staticmethod
    def compute_kappa_quantum(model: HamiltonianNeuralNetwork, hbar: float = ThermodynamicConfig.HBAR) -> float:
        """
        κ cuántico: número de condición con regularización cuántica.
        FIX: Usa método iterativo de potencia en lugar de matriz densa.
        """
        flat_params = torch.cat([p.flatten() for p in model.parameters()])
        n = flat_params.numel()
        
        if n < 2:
            return 1.0
        
        # Para n grande, no podemos construir la matriz de covarianza explícita
        # Usamos el método de potencia para estimar eigenvalores extremos
        if n > Config.KAPPA_MAX_DIM:
            return CrystallographyMetrics._compute_kappa_iterative(flat_params, hbar, n)
        
        # Solo para n pequeño, método directo
        params_centered = flat_params - flat_params.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + hbar * torch.eye(n, device=flat_params.device)
        
        try:
            eigenvals = torch.linalg.eigvalsh(cov_matrix)
            eigenvals = eigenvals[eigenvals > hbar]
            return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
        except:
            return 1.0
    
    @staticmethod
    def _compute_kappa_iterative(params: torch.Tensor, hbar: float, n: int, 
                                  max_iters: int = 100, tol: float = 1e-6) -> float:
        """
        Método de potencia para estimar κ sin construir matriz.
        Estima λ_max y λ_min de la matriz de covarianza regularizada.
        """
        device = params.device
        
        # Centrar parámetros
        params_centered = params - params.mean()
        
        # Estimación de λ_max usando método de potencia
        v_max = torch.randn(n, device=device)
        v_max = v_max / torch.norm(v_max)
        
        for _ in range(max_iters):
            # Multiplicación matriz-vector implícita: (X^T X / n + hbar*I) v
            # donde X es params_centered (vector), así que X^T X es outer product
            v_new = (params_centered * (params_centered @ v_max) / n) + hbar * v_max
            v_new_norm = torch.norm(v_new)
            
            if v_new_norm < tol:
                break
                
            v_max = v_new / v_new_norm
        
        lambda_max = (params_centered * (params_centered @ v_max) / n + hbar * v_max) @ v_max
        
        # Para λ_min, usamos la matriz inversa o shift-and-invert
        # Aproximación: λ_min ≈ hbar para matriz bien condicionada
        # O usamos método de potencia inverso con regularización
        v_min = torch.randn(n, device=device)
        v_min = v_min / torch.norm(v_min)
        
        # Shift para encontrar el mínimo: aplicamos a (A - λ_max*I)^-1
        # Aproximación iterativa simple
        for _ in range(max_iters // 2):  # Menos iteraciones para mínimo
            # (A + shift*I)^-1 v ≈ v / (λ + shift) para aproximación rugosa
            # Usamos gradiente descendente para resolver sistema
            Av = (params_centered * (params_centered @ v_min) / n) + hbar * v_min
            # Aproximación: inversa diagonal
            v_new = v_min - 0.1 * (Av - v_min * (v_min @ Av))
            v_new_norm = torch.norm(v_new)
            
            if v_new_norm < tol:
                break
                
            v_min = v_new / v_new_norm
        
        lambda_min = (params_centered * (params_centered @ v_min) / n + hbar * v_min) @ v_min
        lambda_min = torch.clamp(lambda_min, min=hbar)
        
        kappa = lambda_max / lambda_min
        return kappa.item() if kappa.isfinite() else float(n)  # Fallback: dimensión como cota


    @staticmethod
    def compute_poynting_vector(model: HamiltonianNeuralNetwork) -> Dict[str, Any]:
        """
        Vector de Poynting: flujo de energía en el espacio de parámetros.
        Análogo electromagnético para redes neuronales.
        """
        # Campo "eléctrico": gradientes de pesos
        E = torch.cat([p.flatten() for p in model.parameters()])
        
        # Campo "magnético": no localidad (diferencias entre capas adyacentes)
        spectral_norms = []
        for layer in model.spectral_layers:
            # Verificar si la capa tiene atributo 'weight' antes de acceder
            if hasattr(layer, 'weight') and layer.weight is not None:
                spectral_norms.append(torch.norm(layer.weight.data))
            elif hasattr(layer, 'conv') and layer.conv is not None:
                # Para capas convolucionales dentro de SpectralLayer
                spectral_norms.append(torch.norm(layer.conv.weight.data))
            elif hasattr(layer, 'linear') and layer.linear is not None:
                # Para capas lineales dentro de SpectralLayer
                spectral_norms.append(torch.norm(layer.linear.weight.data))
            else:
                # Si no se puede obtener un peso representativo, usar norma de parámetros
                layer_params = torch.cat([p.flatten() for p in layer.parameters()])
                if layer_params.numel() > 0:
                    spectral_norms.append(torch.norm(layer_params))
                else:
                    spectral_norms.append(torch.tensor(0.0))
        
        if len(spectral_norms) > 1:
            # Diferencias entre capas consecutivas
            H_magnitude = sum(abs(spectral_norms[i] - spectral_norms[i+1])
                            for i in range(len(spectral_norms)-1))
        else:
            H_magnitude = torch.tensor(0.0)
        
        # Poynting ~ E × H
        poynting_magnitude = torch.norm(E) * H_magnitude * ThermodynamicConfig.ENERGY_FLOW_SCALE
        
        # Distribución de energía por componente
        energy_distribution = {
            'encoder_flow': torch.norm(model.input_transform.weight.data).item() if hasattr(model, 'input_transform') and hasattr(model.input_transform, 'weight') else 0.0,
            'decoder_flow': torch.norm(model.output_transform.weight.data).item() if hasattr(model, 'output_transform') and hasattr(model.output_transform, 'weight') else 0.0,
            'spectral_total_flow': sum(spectral_norms).item() if spectral_norms else 0.0
        }
        
        return {
            'poynting_magnitude': poynting_magnitude.item(),
            'energy_distribution': energy_distribution,
            'is_radiating': poynting_magnitude.item() > ThermodynamicConfig.POYNTING_THRESHOLD,
            'field_orthogonality': H_magnitude.item() if isinstance(H_magnitude, torch.Tensor) else H_magnitude
        }


    @staticmethod
    def compute_all_metrics(model: HamiltonianNeuralNetwork, 
                           val_x: torch.Tensor, 
                           val_y: torch.Tensor) -> Dict[str, Any]:
        """Calcula todas las métricas cristalográficas."""
        
        delta = CrystallographyMetrics.compute_discretization_margin(model)
        alpha = CrystallographyMetrics.compute_alpha_purity(model)
        
        metrics = {
            'kappa': CrystallographyMetrics.compute_kappa(model, val_x, val_y),
            'delta': delta,
            'alpha': alpha,
            'kappa_q': CrystallographyMetrics.compute_kappa_quantum(model),
            'lc': CrystallographyMetrics.compute_local_complexity(model),
            'poynting': CrystallographyMetrics.compute_poynting_vector(model)
        }
        
        # Métricas derivadas
        metrics['purity_index'] = 1.0 - delta
        metrics['is_crystal'] = alpha > ThermodynamicConfig.CRYSTAL_ALPHA_THRESHOLD
        metrics['energy_flow'] = metrics['poynting']['poynting_magnitude']
        
        return metrics

# ============================================================================
# MÉTRICAS TERMODINÁMICAS (adaptadas de Strassen)
# ============================================================================

class ThermodynamicMetrics:
    """
    Análisis termodinámico del proceso de entrenamiento.
    Temperatura efectiva, calor específico, transiciones de fase.
    """
    
    @staticmethod
    def compute_effective_temperature(gradient_buffer: List[torch.Tensor], 
                                     learning_rate: float) -> float:
        """
        T_eff = (lr/2) * Var(∇L). Temperatura de fluctuaciones.
        """
        if len(gradient_buffer) < 2:
            return 0.0
        
        grads = torch.stack([g.flatten() for g in gradient_buffer])
        
        second_moment = torch.mean(torch.norm(grads, dim=1)**2)
        first_moment_sq = torch.norm(torch.mean(grads, dim=0))**2
        variance = second_moment - first_moment_sq
        
        return float((learning_rate / 2.0) * variance)
    
    @staticmethod
    def compute_specific_heat(loss_history: List[float], 
                             temp_history: List[float],
                             cv_threshold: float = ThermodynamicConfig.CV_CRITICAL_THRESHOLD) -> Tuple[float, bool]:
        """
        C_v = Var(U) / T^2. Detecta transiciones de fase (picos en C_v).
        """
        if len(loss_history) < 2 or len(temp_history) < 2:
            return 0.0, False
        
        u_var = torch.tensor(loss_history).var()
        t_mean = torch.tensor(temp_history).mean()
        
        cv = u_var / (t_mean**2 + 1e-10)
        is_latent_crystallization = cv > cv_threshold
        
        return float(cv), is_latent_crystallization
    
    @staticmethod
    def compute_critical_exponents(temp_history: List[float], 
                                  cv_history: List[float],
                                  alpha_history: List[float]) -> Dict[str, float]:
        """
        Exponentes críticos cerca de transiciones de fase.
        """
        if len(temp_history) < 5 or len(cv_history) < 5:
            return {
                'alpha_exponent': 0.0,
                'nu_exponent': 0.0,
                'z_exponent': 0.0,
                'critical_temperature': 0.0
            }
        
        cv_array = np.array(cv_history)
        temp_array = np.array(temp_history)
        
        if len(cv_array) == 0 or np.all(cv_array == 0):
            return {
                'alpha_exponent': 0.0,
                'nu_exponent': 0.0,
                'z_exponent': 0.0,
                'critical_temperature': 0.0
            }
        
        # Temperatura crítica (máximo de C_v)
        critical_idx = np.argmax(cv_array)
        T_c = temp_array[critical_idx]
        
        # Exponente α: C_v ~ |T - T_c|^{-α}
        delta_T = np.abs(temp_array - T_c) + ThermodynamicConfig.ENTROPY_EPS
        log_delta_T = np.log(delta_T)
        log_cv = np.log(cv_array + ThermodynamicConfig.ENTROPY_EPS)
        
        near_critical = delta_T < (0.2 * T_c) if T_c > 0 else np.ones_like(delta_T, dtype=bool)
        
        alpha_exp = 0.0
        if np.sum(near_critical) > 2:
            try:
                alpha_exp, _ = np.polyfit(log_delta_T[near_critical], log_cv[near_critical], 1)
                alpha_exp = -float(alpha_exp)
            except:
                pass
        
        # Exponente ν
        nu_exp = 0.0
        if len(alpha_history) > 2:
            alpha_array = np.array(alpha_history)
            correlation_length = 1.0 / (alpha_array + ThermodynamicConfig.ENTROPY_EPS)
            log_xi = np.log(correlation_length + ThermodynamicConfig.ENTROPY_EPS)
            
            if np.sum(near_critical) > 2:
                try:
                    nu_exp, _ = np.polyfit(log_delta_T[near_critical], log_xi[near_critical], 1)
                    nu_exp = -float(nu_exp)
                except:
                    pass
        
        z_exp = alpha_exp / nu_exp if nu_exp > ThermodynamicConfig.ENTROPY_EPS else 0.0
        
        return {
            'alpha_exponent': alpha_exp,
            'nu_exponent': nu_exp,
            'z_exponent': z_exp,
            'critical_temperature': float(T_c)
        }
    
    @staticmethod
    def compute_equation_of_state(temp_eff: float, 
                                 alpha: float, 
                                 kappa: float) -> Dict[str, Any]:
        """
        Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
        Relación constitutiva cristal-vidrio.
        """
        T_0 = ThermodynamicConfig.T_0
        c = ThermodynamicConfig.C_COEFF
        
        T_critical_predicted = T_0 * np.exp(-c * alpha)
        delta_T = temp_eff - T_critical_predicted
        
        # Clasificación de fase
        if delta_T < -ThermodynamicConfig.MIN_VARIANCE_THRESHOLD:
            phase = "subcritical_crystal"
        elif abs(delta_T) < ThermodynamicConfig.MIN_VARIANCE_THRESHOLD:
            phase = "critical_point"
        else:
            phase = "supercritical_glass"
        
        # Presión algorítmica
        pressure = alpha * kappa
        
        return {
            'temperature_effective': float(temp_eff),
            'temperature_critical': float(T_critical_predicted),
            'deviation_from_equilibrium': float(delta_T),
            'phase_classification': phase,
            'algorithmic_pressure': float(pressure),
            'equation_form': f'T_c(α) = {T_0} * exp(-{c} * α)',
            'is_equilibrium': abs(delta_T) < ThermodynamicConfig.MIN_VARIANCE_THRESHOLD
        }
    
    @staticmethod
    def compute_mutual_information(weights: torch.Tensor, 
                                  gradients: torch.Tensor) -> float:
        """Información mutua pesos-gradientes."""
        w_flat = weights.flatten()
        g_flat = gradients.flatten()
        
        w_std = torch.std(w_flat)
        g_std = torch.std(g_flat)
        
        if w_std == 0 or g_std == 0:
            return 0.0
        
        # Correlación como proxy de información mutua
        correlation = torch.corrcoef(torch.stack([w_flat, g_flat]))[0, 1]
        
        if torch.isnan(correlation) or abs(correlation) >= 1:
            return 0.0
        
        mi = -0.5 * torch.log(1 - correlation**2 + 1e-10)
        return float(mi)
    
    @staticmethod
    def estimate_hbar_algorithmic(model_complexity: float, 
                                 weight_dim: int, 
                                 mutual_information: float) -> float:
        """ħ algorítmico efectivo."""
        if weight_dim == 0 or mutual_information == 0:
            return ThermodynamicConfig.HBAR
        
        hbar_alg = model_complexity / (weight_dim * mutual_information)
        return float(hbar_alg)
    
    @staticmethod
    def compute_fisher_information_matrix(model: HamiltonianNeuralNetwork,
                                         samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Matriz de información de Fisher."""
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        fisher = torch.zeros(n_params, n_params, device=next(model.parameters()).device)
        
        for x, y in samples:
            model.zero_grad()
            output = model(x)
            log_prob = -F.mse_loss(output, y)
            
            grads = torch.autograd.grad(log_prob, model.parameters(), create_graph=True)
            grad_vector = torch.cat([g.flatten() for g in grads])
            
            fisher += torch.outer(grad_vector, grad_vector)
        
        fisher /= len(samples)
        return fisher
    
    @staticmethod
    def compute_ricci_curvature(fisher_matrix: torch.Tensor) -> float:
        """Curvatura de Ricci escalar."""
        try:
            eigenvalues = torch.linalg.eigvalsh(fisher_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) < 2:
                return 0.0
            
            ricci_scalar = torch.sum(1.0 / (eigenvalues + 1e-10))
            return float(ricci_scalar)
        except:
            return 0.0
    
    @staticmethod
    def calculate_carnot_efficiency(delta_alpha: float, 
                                   total_flops: float, 
                                   initial_alpha: float = 0.0) -> Dict[str, Any]:
        """Eficiencia de Carnot del proceso de aprendizaje."""
        if total_flops <= 0:
            return {'efficiency': 0.0, 'work_done': 0.0, 'heat_dissipated': 0.0}
        
        invariance_gain = delta_alpha - initial_alpha
        work_done = invariance_gain
        heat_dissipated = total_flops * 1e-9
        
        efficiency = work_done / (heat_dissipated + 1e-10)
        carnot_limit = 1.0 - (initial_alpha + 1e-3) / (delta_alpha + 1e-3)
        
        return {
            'efficiency': float(efficiency),
            'work_done': float(work_done),
            'heat_dissipated': float(heat_dissipated),
            'carnot_limit': float(carnot_limit),
            'relative_efficiency': float(efficiency / (carnot_limit + 1e-10))
        }

# ============================================================================
# ESPECTROSCOPÍA (adaptada de Strassen)
# ============================================================================

class SpectroscopyMetrics:
    """
    Análisis espectroscópico de pesos: difracción, descomposición, parámetros de red.
    """
    
    @staticmethod
    def compute_weight_diffraction(model: HamiltonianNeuralNetwork) -> Dict[str, Any]:
        """
        Patrón de difracción de pesos (FFT).
        Detecta periodicidad cristalina (picos de Bragg).
        """
        # Concatenar todos los pesos
        all_weights = torch.cat([p.flatten() for p in model.parameters()])
        # FFT
        fft_spectrum = torch.fft.fft(all_weights)
        power_spectrum = torch.abs(fft_spectrum)**2
        # Detectar picos (Bragg)
        threshold = torch.mean(power_spectrum) + 2 * torch.std(power_spectrum)
        peaks = []
        for i, power in enumerate(power_spectrum):
            if power > threshold:
                peaks.append({
                    'frequency': i,
                    'intensity': float(power),
                    'normalized_intensity': float(power / power_spectrum.max())
                })
        # Estructura cristalina si hay picos bien definidos
        is_crystalline = (0 < len(peaks) < len(power_spectrum) // 4)
        return {
            'power_spectrum': power_spectrum.detach().cpu().numpy().tolist(),
            'bragg_peaks': peaks,
            'n_peaks': len(peaks),
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': float(SpectroscopyMetrics._compute_spectral_entropy(power_spectrum)),
            'dominant_frequency': int(torch.argmax(power_spectrum).item()) if len(power_spectrum) > 0 else 0
        }
        
    @staticmethod
    def _compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
        """Entropía espectral de Shannon."""
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy)
    
    @staticmethod
    def extract_lattice_parameters(weight_tensor: torch.Tensor, rank: int = 7) -> Dict[str, Any]:
        """
        Extrae parámetros de red vía SVD.
        """
        # Asegurar 2D
        if weight_tensor.dim() == 1:
            total_size = weight_tensor.numel()
            side_dim = int(np.ceil(np.sqrt(total_size)))
            padded_size = side_dim * side_dim
            if total_size < padded_size:
                padding = torch.zeros(padded_size - total_size, device=weight_tensor.device)
                weight_tensor = torch.cat([weight_tensor, padding])
            weight_tensor = weight_tensor[:side_dim * side_dim].reshape(side_dim, side_dim)
        
        try:
            U, S, Vh = torch.linalg.svd(weight_tensor, full_matrices=False)
            
            # Umbral de ruido térmico
            threshold = S[0] * ThermodynamicConfig.MIN_VARIANCE_THRESHOLD ** 0.5
            S_clean = torch.where(S > threshold, S, torch.zeros_like(S))
            
            # Rango efectivo
            rank_effective = torch.sum(S_clean > threshold).item()
            
            # Gap espectral
            spectral_gap = (S[0] / (S[1] + ThermodynamicConfig.ENTROPY_EPS)).item() if len(S) > 1 else float('inf')
            
            # Reconstrucción
            rank_truncated = min(rank, len(S_clean))
            U_truncated = U[:, :rank_truncated]
            S_truncated = S_clean[:rank_truncated]
            Vh_truncated = Vh[:rank_truncated, :]
            
            clean_crystal = U_truncated @ torch.diag(S_truncated) @ Vh_truncated
            reconstruction_error = torch.norm(weight_tensor - clean_crystal) / (torch.norm(weight_tensor) + ThermodynamicConfig.ENTROPY_EPS)
            
            return {
                'singular_values': S.cpu().numpy().tolist(),
                'clean_singular_values': S_truncated.cpu().numpy().tolist(),
                'effective_rank': int(rank_effective),
                'spectral_gap': float(spectral_gap),
                'reconstruction_error': float(reconstruction_error),
                'thermal_noise_threshold': float(threshold),
                'rank_truncated': rank_truncated
            }
        except Exception as e:
            return {
                'error': str(e),
                'tensor_shape': list(weight_tensor.shape)
            }
    
    @staticmethod
    def compute_gibbs_free_energy(loss: float, temp: float, entropy: float) -> float:
        """Energía libre de Gibbs."""
        return loss + (temp * entropy)

# ============================================================================
# VERIFICADOR DE CHECKPOINTS MEJORADO
# ============================================================================

class CheckpointVerifier:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        # Cargar checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Recrear modelo
        self.model = HamiltonianNeuralNetwork(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(self.device)
        
        # Cargar pesos
        if 'model_state_dict' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(self.checkpoint)
        
        # Dataset para validación
        self.dataset = HamiltonianDataset(
            num_samples=Config.NUM_SAMPLES,
            grid_size=Config.GRID_SIZE
        )
        self.val_x, self.val_y = self.dataset.get_validation_batch()
        self.val_x = self.val_x.to(self.device)
        self.val_y = self.val_y.to(self.device)
        
        # Extraer info del checkpoint
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.stored_metrics = self.checkpoint.get('metrics', {})
        self.stored_val_acc = self.checkpoint.get('val_acc', None)
        self.config = self.checkpoint.get('config', {})
        
        # Buffer para métricas termodinámicas
        self.gradient_buffer = deque(maxlen=50)
        self.loss_history = deque(maxlen=100)
        self.temp_history = deque(maxlen=100)
        self.cv_history = deque(maxlen=100)
        
    def verify_all_metrics(self) -> Dict[str, Any]:
        """Calcula TODAS las métricas desde cero y compara con las guardadas"""
        
        print(f"\n{'='*70}")
        print(f"VERIFICANDO CHECKPOINT: {self.checkpoint_path}")
        print(f"Epoch reportada: {self.epoch}")
        print(f"{'='*70}\n")
        
        results = {
            'checkpoint_path': self.checkpoint_path,
            'epoch_reported': self.epoch,
            'file_size_mb': os.path.getsize(self.checkpoint_path) / (1024*1024),
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Verificar integridad de pesos (NaN/Inf)
        weight_integrity = self._check_weight_integrity()
        results['weight_integrity'] = weight_integrity
        
        # 2. Métricas de validación básicas
        val_metrics = self._compute_validation_metrics()
        results['validation_metrics'] = val_metrics
        
        # 3. Métricas de discretización (delta, alpha, etc)
        discret_metrics = self._compute_discretization_metrics()
        results['discretization_metrics'] = discret_metrics
        
        # 4. Métricas de cuantización
        quant_metrics = self._compute_quantization_metrics()
        results['quantization_metrics'] = quant_metrics
        
        # 5. Loss completo
        loss_metrics = self._compute_loss_metrics()
        results['loss_metrics'] = loss_metrics
        
        # 6. NUEVO: Métricas cristalográficas completas
        crystal_metrics = self._compute_crystallography_metrics()
        results['crystallography_metrics'] = crystal_metrics
        
        # 7. NUEVO: Métricas termodinámicas
        thermo_metrics = self._compute_thermodynamic_metrics()
        results['thermodynamic_metrics'] = thermo_metrics
        
        # 8. NUEVO: Espectroscopía
        spectroscopy = self._compute_spectroscopy()
        results['spectroscopy'] = spectroscopy
        
        # 9. Comparar con métricas almacenadas
        comparison = self._compare_with_stored(results)
        results['comparison_with_stored'] = comparison
        
        # 10. Verificar consistencia interna
        consistency = self._check_internal_consistency(results)
        results['consistency_checks'] = consistency
        
        # 11. Potencial termodinámico
        potential = self._compute_thermodynamic_potential(results)
        results['thermodynamic_potential'] = potential
        
        # 12. Resumen de salud
        health = self._compute_health_score(results)
        results['health_score'] = health
        
        self._print_report(results)
        
        return results
    
    def _check_weight_integrity(self) -> Dict[str, Any]:
        """
        Verifica que los pesos no tengan NaN/Inf.
        FIX: Evita calcular std en tensores con 1 elemento.
        """
        has_nan = False
        has_inf = False
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        param_stats = {}
        
        for name, param in self.model.named_parameters():
            data = param.data
            total_params += data.numel()
            
            layer_nan = torch.isnan(data).sum().item()
            layer_inf = torch.isinf(data).sum().item()
            
            if layer_nan > 0:
                has_nan = True
                nan_params += layer_nan
            if layer_inf > 0:
                has_inf = True
                inf_params += layer_inf
            
            # FIX: std() requiere al menos 2 elementos para corrección de sesgo
            numel = data.numel()
            if numel == 0:
                std_val = 0.0
                mean_val = 0.0
                min_val = 0.0
                max_val = 0.0
            elif numel == 1:
                std_val = 0.0  # Std de un solo valor es 0
                mean_val = data.item()
                min_val = data.item()
                max_val = data.item()
            else:
                std_val = data.std().item()
                mean_val = data.mean().item()
                min_val = data.min().item()
                max_val = data.max().item()
            
            param_stats[name] = {
                'shape': list(data.shape),
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'has_nan': layer_nan > 0,
                'has_inf': layer_inf > 0
            }
        
        return {
            'is_valid': not (has_nan or has_inf),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'total_params': total_params,
            'nan_params': nan_params,
            'inf_params': inf_params,
            'corruption_ratio': (nan_params + inf_params) / total_params if total_params > 0 else 0,
            'layer_stats': param_stats
        }

    def _compute_validation_metrics(self) -> Dict[str, float]:
        """Calcula MSE y accuracy de validación desde cero"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.val_x)
            
            # MSE total
            mse = F.mse_loss(outputs, self.val_y).item()
            
            # MSE por sample para accuracy
            mse_per_sample = ((outputs - self.val_y) ** 2).mean(dim=(1, 2))
            accuracy = (mse_per_sample < Config.MSE_THRESHOLD).float().mean().item()
            
            # Máximo error
            max_error = (outputs - self.val_y).abs().max().item()
            
            # Error por coordenada
            errors_per_coord = (outputs - self.val_y).abs().mean(dim=0).cpu().numpy()
        
        return {
            'mse': mse,
            'accuracy': accuracy,
            'max_error': max_error,
            'mean_error_per_coord': errors_per_coord.tolist(),
            'threshold_used': Config.MSE_THRESHOLD
        }
    
    def _compute_discretization_metrics(self) -> Dict[str, Any]:
        """Calcula delta, alpha, purity, etc."""
        all_deltas = []
        layer_deltas = {}
        
        for name, param in self.model.named_parameters():
            if param.numel() > 0:
                rounded = torch.round(param.data)
                delta_layer = (param.data - rounded).abs().max().item()
                layer_deltas[name] = delta_layer
                all_deltas.append(delta_layer)
        
        delta_max = max(all_deltas) if all_deltas else 1.0
        delta_mean = np.mean(all_deltas) if all_deltas else 1.0
        
        alpha = -np.log(delta_max + 1e-15) if delta_max > 1e-10 else 20.0
        purity = 1.0 - delta_max
        is_crystal = alpha > ThermodynamicConfig.CRYSTAL_ALPHA_THRESHOLD
        
        # Distribución de pesos
        all_weights = torch.cat([p.data.flatten() for p in self.model.parameters()])
        
        near_zero = (all_weights.abs() < 0.1).sum().item()
        near_one = ((all_weights - 1).abs() < 0.1).sum().item()
        near_minus_one = ((all_weights + 1).abs() < 0.1).sum().item()
        near_integer = ((all_weights - torch.round(all_weights)).abs() < 0.1).sum().item()
        total = all_weights.numel()
        
        return {
            'delta_max': delta_max,
            'delta_mean': delta_mean,
            'alpha': alpha,
            'purity_index': purity,
            'is_crystal': is_crystal,
            'layer_deltas': layer_deltas,
            'weight_distribution': {
                'near_zero': near_zero / total,
                'near_one': near_one / total,
                'near_minus_one': near_minus_one / total,
                'near_any_integer': near_integer / total,
                'total_params': total
            }
        }
    
    def _compute_quantization_metrics(self) -> Dict[str, float]:
        """Calcula la penalización de cuantización"""
        penalty = 0.0
        total_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad and param.numel() > 0:
                rounded = torch.round(param.data)
                penalty += torch.sum(torch.abs(param.data - rounded)).item()
                total_params += param.numel()
        
        quant_penalty = penalty / total_params if total_params > 0 else 0
        
        return {
            'quantization_penalty': quant_penalty,
            'total_params': total_params,
            'sum_abs_errors': penalty
        }
    
    def _compute_loss_metrics(self) -> Dict[str, float]:
        """Reconstruye el loss total"""
        val_metrics = self._compute_validation_metrics()
        quant_metrics = self._compute_quantization_metrics()
        
        mse = val_metrics['mse']
        quant = quant_metrics['quantization_penalty']
        
        lambda_quant = self.config.get('quant_lambda', 0.5)
        total_loss = mse + float(lambda_quant) * quant
        
        return {
            'mse': mse,
            'quantization_penalty': quant,
            'lambda_used': lambda_quant,
            'total_loss': total_loss,
            'loss_decomposition': {
                'mse_ratio': mse / total_loss if total_loss > 0 else 0,
                'quant_ratio': (lambda_quant * quant) / total_loss if total_loss > 0 else 0
            }
        }
    
    # =========================================================================
    # NUEVOS MÉTODOS: Métricas avanzadas del sistema Strassen
    # =========================================================================
    
    def _compute_crystallography_metrics(self) -> Dict[str, Any]:
        """Métricas cristalográficas completas"""
        metrics = CrystallographyMetrics.compute_all_metrics(
            self.model, self.val_x, self.val_y
        )
        return metrics
    
    def _compute_thermodynamic_metrics(self) -> Dict[str, Any]:
        """Métricas termodinámicas"""
        # Simular buffer de gradientes para estimar temperatura
        gradient_buffer = []
        loss_values = []
        
        self.model.train()
        for _ in range(10):  # 10 batches simulados
            self.model.zero_grad()
            outputs = self.model(self.val_x)
            loss = F.mse_loss(outputs, self.val_y)
            loss.backward()
            
            grad = torch.cat([p.grad.flatten() for p in self.model.parameters() 
                            if p.grad is not None]).detach()
            gradient_buffer.append(grad)
            loss_values.append(loss.item())
        
        # Temperatura efectiva
        lr = self.config.get('learning_rate', 0.001)
        t_eff = ThermodynamicMetrics.compute_effective_temperature(gradient_buffer, lr)
        
        # Calor específico
        temp_history = [t_eff] * len(loss_values)
        cv, is_crystallizing = ThermodynamicMetrics.compute_specific_heat(
            loss_values, temp_history
        )
        
        # Exponentes críticos (necesitamos historia temporal real para esto)
        # Aquí usamos valores actuales como aproximación
        critical_exponents = {
            'alpha_exponent': 0.0,  # Requiere serie temporal
            'nu_exponent': 0.0,
            'z_exponent': 0.0,
            'critical_temperature': t_eff if is_crystallizing else 0.0
        }
        
        # Ecuación de estado
        alpha = CrystallographyMetrics.compute_alpha_purity(self.model)
        kappa = CrystallographyMetrics.compute_kappa(self.model, self.val_x, self.val_y)
        equation_of_state = ThermodynamicMetrics.compute_equation_of_state(
            t_eff, alpha, kappa
        )
        
        # Información mutua
        all_weights = torch.cat([p.flatten() for p in self.model.parameters()])
        avg_gradient = torch.mean(torch.stack(gradient_buffer), dim=0)
        mi = ThermodynamicMetrics.compute_mutual_information(all_weights, avg_gradient)
        
        # ħ algorítmico
        model_complexity = -np.log(
            CrystallographyMetrics.compute_discretization_margin(self.model) + 1e-10
        )
        weight_dim = all_weights.numel()
        hbar_alg = ThermodynamicMetrics.estimate_hbar_algorithmic(
            model_complexity, weight_dim, mi
        )
        
        # Curvatura de Ricci (aproximada)
        # Para HPU Core, usamos una aproximación basada en la Hessiana diagonal
        ricci = self._approximate_ricci_curvature()
        
        return {
            'effective_temperature': t_eff,
            'specific_heat_cv': cv,
            'is_crystallizing': is_crystallizing,
            'critical_exponents': critical_exponents,
            'equation_of_state': equation_of_state,
            'mutual_information': mi,
            'hbar_algorithmic': hbar_alg,
            'ricci_curvature': ricci,
            'phase_classification': equation_of_state['phase_classification'],
            'algorithmic_pressure': equation_of_state['algorithmic_pressure']
        }
    
    def _approximate_ricci_curvature(self) -> float:
        """Aproximación de curvatura de Ricci para HPU Core"""
        # Usamos la varianza de los gradientes como proxy de curvatura
        try:
            self.model.zero_grad()
            outputs = self.model(self.val_x)
            loss = F.mse_loss(outputs, self.val_y)
            loss.backward()
            
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads:
                return 0.0
            
            grad_norms = [torch.norm(g).item() for g in grads]
            variance = np.var(grad_norms)
            
            # Curvatura inversamente proporcional a la varianza de gradientes
            return 1.0 / (variance + 1e-6)
        except:
            return 0.0
    
    def _compute_spectroscopy(self) -> Dict[str, Any]:
        """Análisis espectroscópico"""
        # Difracción de pesos
        diffraction = SpectroscopyMetrics.compute_weight_diffraction(self.model)
        # Parámetros de red del input_transform (más interpretable)
        if hasattr(self.model, 'input_transform') and hasattr(self.model.input_transform, 'weight'):
            transform_weights = self.model.input_transform.weight.data
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'weight'):
            transform_weights = self.model.encoder.weight.data
        else:
            # Si no hay transformaciones específicas, usar pesos del primer layer
            first_param = next(iter(self.model.parameters()))
            transform_weights = first_param.data
        
        lattice_params = SpectroscopyMetrics.extract_lattice_parameters(
            transform_weights.flatten(), rank=7
        )
        # Energía libre de Gibbs
        loss = self._compute_loss_metrics()['total_loss']
        temp = self._compute_thermodynamic_metrics()['effective_temperature']
        entropy = -self._compute_discretization_metrics()['alpha']
        gibbs = SpectroscopyMetrics.compute_gibbs_free_energy(loss, temp, entropy)
        return {
            'weight_diffraction': diffraction,
            'transform_lattice_params': lattice_params,
            'gibbs_free_energy': gibbs,
            'is_stable': gibbs < 0,
            'spectral_entropy': diffraction['spectral_entropy']
        }
        
    def _compute_thermodynamic_potential(self, results: Dict) -> Dict[str, Any]:
        """Calcula potencial termodinámico completo"""
        loss_metrics = results['loss_metrics']
        thermo = results['thermodynamic_metrics']
        crystal = results['crystallography_metrics']
        
        potential = ThermodynamicPotential(
            internal_energy=loss_metrics['total_loss'],
            temperature=thermo['effective_temperature'],
            entropy=thermo.get('specific_heat_cv', 0),  # Proxy
            chemical_potential=thermo['mutual_information'],
            crystallinity=crystal['alpha'],
            particle_number=float(sum(p.numel() for p in self.model.parameters()))
        )
        
        return {
            'helmholtz_free_energy': potential.helmholtz_free_energy(),
            'gibbs_free_energy': potential.gibbs_free_energy(),
            'is_stable': potential.is_stable(),
            'internal_energy': potential.internal_energy,
            'temperature': potential.temperature,
            'crystallinity': potential.crystallinity
        }
    
    def _compare_with_stored(self, computed: Dict) -> Dict[str, Any]:
        """Compara métricas calculadas vs almacenadas en el checkpoint"""
        comparisons = {}
        
        mappings = {
            'delta': ('discretization_metrics', 'delta_max'),
            'alpha': ('discretization_metrics', 'alpha'),
            'val_acc': ('validation_metrics', 'accuracy'),
            'val_mse': ('validation_metrics', 'mse'),
            'quant_penalty': ('quantization_metrics', 'quantization_penalty'),
            'total_loss': ('loss_metrics', 'total_loss'),
            'kappa': ('crystallography_metrics', 'kappa'),
            'purity_index': ('discretization_metrics', 'purity_index')
        }
        
        for stored_key, (section, computed_key) in mappings.items():
            stored_val = None
            if stored_key in self.stored_metrics:
                stored_val = self.stored_metrics[stored_key]
            elif stored_key == 'val_acc' and self.stored_val_acc is not None:
                stored_val = self.stored_val_acc
            
            computed_val = computed.get(section, {}).get(computed_key)
            
            if stored_val is not None and computed_val is not None:
                diff = abs(stored_val - computed_val)
                rel_diff = diff / abs(stored_val) if stored_val != 0 else float('inf')
                
                comparisons[stored_key] = {
                    'stored': stored_val,
                    'computed': computed_val,
                    'absolute_diff': diff,
                    'relative_diff': rel_diff,
                    'match': diff < 1e-5
                }
        
        return comparisons
    
    def _check_internal_consistency(self, results: Dict) -> Dict[str, Any]:
        """Verifica consistencia entre métricas relacionadas"""
        checks = {}
        
        # 1. Consistencia delta-alpha
        delta = results['discretization_metrics']['delta_max']
        alpha = results['discretization_metrics']['alpha']
        expected_alpha = -np.log(delta + 1e-15) if delta > 1e-10 else 20.0
        checks['delta_alpha_consistency'] = {
            'expected_alpha': expected_alpha,
            'actual_alpha': alpha,
            'match': abs(expected_alpha - alpha) < 1e-5
        }
        
        # 2. Consistencia loss
        loss_data = results['loss_metrics']
        reconstructed = loss_data['mse'] + loss_data['lambda_used'] * loss_data['quantization_penalty']
        checks['loss_reconstruction'] = {
            'reconstructed': reconstructed,
            'reported': loss_data['total_loss'],
            'match': abs(reconstructed - loss_data['total_loss']) < 1e-5
        }
        
        # 3. Consistencia cristalografía
        crystal = results['crystallography_metrics']
        discret = results['discretization_metrics']
        checks['crystal_metrics_consistency'] = {
            'alpha_match': abs(crystal['alpha'] - discret['alpha']) < 1e-5,
            'delta_match': abs(crystal['delta'] - discret['delta_max']) < 1e-5,
            'is_crystal_match': crystal['is_crystal'] == discret['is_crystal']
        }
        
        # 4. Consistencia termodinámica
        thermo = results['thermodynamic_metrics']
        checks['thermodynamic_consistency'] = {
            'positive_temperature': thermo['effective_temperature'] >= 0,
            'stable_phase_defined': thermo['phase_classification'] != 'unknown'
        }
        
        return checks
    
    def _compute_health_score(self, results: Dict) -> Dict[str, Any]:
        """Calcula un score de salud del checkpoint (0-100)"""
        score = 100.0
        issues = []
        
        # Penalizaciones básicas
        if not results['weight_integrity']['is_valid']:
            score -= 50
            issues.append("CORRUPTED_WEIGHTS")
        if results['weight_integrity']['corruption_ratio'] > 0:
            score -= 30 * results['weight_integrity']['corruption_ratio']
            issues.append("PARTIAL_CORRUPTION")
        
        # Comparación con stored
        mismatches = sum(1 for v in results['comparison_with_stored'].values() if not v.get('match', True))
        score -= 10 * mismatches
        if mismatches > 0:
            issues.append(f"METRIC_MISMATCHES:{mismatches}")
        
        # Consistencia interna
        for check_name, check_data in results['consistency_checks'].items():
            if isinstance(check_data, dict) and not check_data.get('match', True):
                score -= 5
                issues.append(f"INCONSISTENT:{check_name}")
        
        # Delta muy alto
        if results['discretization_metrics']['delta_max'] > 0.5:
            score -= 10
            issues.append("HIGH_DELTA")
        
        # Lambda extremo
        lambda_val = results['loss_metrics']['lambda_used']
        if lambda_val > 1e30:
            score -= 15
            issues.append("EXTREME_LAMBDA")
        
        # NUEVO: Penalizaciones termodinámicas
        thermo = results['thermodynamic_metrics']
        if thermo['effective_temperature'] > 1.0:
            score -= 5
            issues.append("HIGH_TEMPERATURE")
        if thermo['phase_classification'] == 'supercritical_glass':
            score -= 10
            issues.append("GLASS_PHASE")
        
        # NUEVO: Penalizaciones cristalográficas
        crystal = results['crystallography_metrics']
        if crystal['kappa'] > 1e6:
            score -= 10
            issues.append("ILL_CONDITIONED_KAPPA")
        if not crystal['is_crystal'] and results['discretization_metrics']['is_crystal']:
            issues.append("CRYSTAL_STATUS_MISMATCH")
        
        # Bonus por estabilidad termodinámica
        potential = results['thermodynamic_potential']
        if potential['is_stable']:
            score += 5
        
        return {
            'score': max(0, min(100, score)),
            'status': 'HEALTHY' if score > 80 else 'DEGRADED' if score > 50 else 'CRITICAL',
            'issues': issues,
            'is_usable': score > 30 and results['weight_integrity']['is_valid'],
            'crystallographic_grade': self._assign_crystallographic_grade(
                results['discretization_metrics']['delta_max'],
                results['discretization_metrics']['alpha']
            )
        }
    
    def _assign_crystallographic_grade(self, delta: float, alpha: float) -> str:
        """Asigna grado cristalográfico"""
        if delta < 0.01:
            return "Optical Crystal (δ<0.01)"
        elif delta < 0.1:
            return "Industrial Crystal (δ<0.1)"
        elif delta < 0.3:
            return "Polycrystalline (δ<0.3)"
        elif delta < 0.5:
            return "Amorphous Glass (δ<0.5)"
        else:
            return "Defective Structure (δ≥0.5)"
    
    def _print_report(self, results: Dict):
        """Imprime reporte formateado con todas las métricas nuevas"""
        print(f"\n{'='*70}")
        print("REPORTE DE VERIFICACIÓN - HPU CRYSTALLOGRAPHY SYSTEM")
        print(f"{'='*70}")
        
        # Integridad
        wi = results['weight_integrity']
        print(f"\n[INTEGRIDAD DE PESOS]")
        print(f"  Válido: {wi['is_valid']} {'✓' if wi['is_valid'] else '✗'}")
        if not wi['is_valid']:
            print(f"  NaN params: {wi['nan_params']:,}")
            print(f"  Inf params: {wi['inf_params']:,}")
            print(f"  Corrupción: {wi['corruption_ratio']:.2%}")
        
        # Métricas principales
        print(f"\n[MÉTRICAS DE VALIDACIÓN]")
        vm = results['validation_metrics']
        print(f"  MSE: {vm['mse']:.6f}")
        print(f"  Accuracy: {vm['accuracy']:.4f}")
        print(f"  Max Error: {vm['max_error']:.6f}")
        
        print(f"\n[MÉTRICAS DE DISCRETIZACIÓN]")
        dm = results['discretization_metrics']
        print(f"  Delta (max): {dm['delta_max']:.6f}")
        print(f"  Delta (mean): {dm['delta_mean']:.6f}")
        print(f"  Alpha: {dm['alpha']:.2f}")
        print(f"  Is Crystal: {dm['is_crystal']} {'✓' if dm['is_crystal'] else '✗'}")
        print(f"  Purity Index: {dm['purity_index']:.4f}")
        
        # NUEVO: Métricas cristalográficas
        print(f"\n[CRISTALOGRAFÍA AVANZADA]")
        cm = results['crystallography_metrics']
        print(f"  κ (condición): {cm['kappa']:.2e}")
        print(f"  κ_q (cuántico): {cm['kappa_q']:.2e}")
        print(f"  LC (complejidad local): {cm['lc']:.4f}")
        print(f"  Flujo de Poynting: {cm['energy_flow']:.2e}")
        print(f"  Radiando: {'✓' if cm['poynting']['is_radiating'] else '✗'}")
        
        # NUEVO: Métricas termodinámicas
        print(f"\n[TERMODINÁMICA]")
        tm = results['thermodynamic_metrics']
        print(f"  T_efectiva: {tm['effective_temperature']:.2e}")
        print(f"  C_v (calor específico): {tm['specific_heat_cv']:.2e}")
        print(f"  Fase: {tm['phase_classification']}")
        print(f"  Transición: {'✓' if tm['is_crystallizing'] else '✗'}")
        print(f"  Presión alg.: {tm['algorithmic_pressure']:.2e}")
        print(f"  ħ_alg: {tm['hbar_algorithmic']:.2e}")
        print(f"  I_mutua: {tm['mutual_information']:.4f}")
        
        # NUEVO: Espectroscopía
        print(f"\n[ESPECTROSCOPÍA]")
        sp = results['spectroscopy']
        print(f"  Entropía espectral: {sp['spectral_entropy']:.4f}")
        print(f"  Picos de Bragg: {sp['weight_diffraction']['n_peaks']}")
        print(f"  Estructura cristalina: {'✓' if sp['weight_diffraction']['is_crystalline_structure'] else '✗'}")
        print(f"  Energía libre G: {sp['gibbs_free_energy']:.4f}")
        print(f"  Estable: {'✓' if sp['is_stable'] else '✗'}")
        
        # Potencial termodinámico
        print(f"\n[POTENCIAL TERMODINÁMICO]")
        pot = results['thermodynamic_potential']
        print(f"  F (Helmholtz): {pot['helmholtz_free_energy']:.4f}")
        print(f"  G (Gibbs): {pot['gibbs_free_energy']:.4f}")
        print(f"  Estable: {'✓' if pot['is_stable'] else '✗'}")
        
        # Comparación
        print(f"\n[COMPARACIÓN CON CHECKPOINT]")
        for key, comp in results['comparison_with_stored'].items():
            status = "✓" if comp['match'] else "✗"
            print(f"  {key}: stored={comp['stored']:.6f}, computed={comp['computed']:.6f} "
                  f"diff={comp['absolute_diff']:.2e} {status}")
        
        # Salud
        health = results['health_score']
        print(f"\n[SCORE DE SALUD]")
        print(f"  Puntuación: {health['score']:.1f}/100")
        print(f"  Estado: {health['status']}")
        print(f"  Grado cristalográfico: {health['crystallographic_grade']}")
        print(f"  Usable: {health['is_usable']} {'✓' if health['is_usable'] else '✗'}")
        if health['issues']:
            print(f"  Problemas: {', '.join(health['issues'])}")
        
        print(f"\n{'='*70}")


# ============================================================================
# FUNCIONES DE UTILIDAD PARA ANÁLISIS POR LOTES
# ============================================================================

def verify_latest_checkpoints(checkpoint_dir: str = "crystal_checkpoints", n: int = 5):
    """Verifica los N checkpoints más recientes con análisis completo"""
    pattern = os.path.join(checkpoint_dir, "*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\nEncontrados {len(checkpoints)} checkpoints")
    print(f"Verificando los {min(n, len(checkpoints))} más recientes...")
    
    results_list = []
    for cp in checkpoints[:n]:
        try:
            verifier = CheckpointVerifier(cp)
            results = verifier.verify_all_metrics()
            results_list.append(results)
        except Exception as e:
            print(f"\nERROR verificando {cp}: {e}")
            results_list.append({
                'checkpoint_path': cp,
                'error': str(e),
                'health_score': {'score': 0, 'status': 'ERROR', 'is_usable': False}
            })
    
    # Resumen final comparativo
    print(f"\n{'='*80}")
    print("RESUMEN COMPARATIVO - ANALISIS CRISTALOGRÁFICO")
    print(f"{'='*80}")
    print(f"{'Epoch':>8} | {'Score':>6} | {'Alpha':>6} | {'T_eff':>10} | {'Fase':>20} | {'Grado':>25}")
    print("-" * 80)
    
    for r in results_list:
        if 'error' in r:
            continue
            
        health = r.get('health_score', {})
        epoch = r.get('epoch_reported', '?')
        score = health.get('score', 0)
        
        crystal = r.get('crystallography_metrics', {})
        alpha = crystal.get('alpha', 0)
        
        thermo = r.get('thermodynamic_metrics', {})
        t_eff = thermo.get('effective_temperature', 0)
        phase = thermo.get('phase_classification', 'unknown')[:18]
        
        grade = health.get('crystallographic_grade', 'Unknown')[:23]
        
        print(f"{epoch:>8} | {score:>6.1f} | {alpha:>6.2f} | {t_eff:>10.2e} | {phase:>20} | {grade:>25}")
    
    # Guardar reporte comparativo
    summary_path = os.path.join(checkpoint_dir, "comparative_analysis.json")
    with open(summary_path, 'w') as f:
        json.dump(results_list, f, indent=2, default=str)
    print(f"\nReporte comparativo guardado en: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='HPU Core Checkpoint Verifier with Crystallography')
    parser.add_argument('checkpoint', nargs='?', default=None, 
                       help='Specific checkpoint to verify (or latest if not provided)')
    parser.add_argument('--latest', type=int, default=None,
                       help='Verify N latest checkpoints')
    parser.add_argument('--dir', default='crystal_checkpoints',
                       help='Directory to search for checkpoints')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparative analysis of all checkpoints')
    
    args = parser.parse_args()
    
    if args.compare:
        verify_latest_checkpoints(args.dir, 10)
    elif args.latest:
        verify_latest_checkpoints(args.dir, args.latest)
    elif args.checkpoint:
        verifier = CheckpointVerifier(args.checkpoint)
        results = verifier.verify_all_metrics()
        
        # Guardar reporte detallado
        report_path = args.checkpoint.replace('.pth', '_verification.json')
        with open(report_path, 'w') as f:
            clean_results = json.loads(json.dumps(results, default=str))
            json.dump(clean_results, f, indent=2)
        print(f"\nReporte detallado guardado en: {report_path}")
    else:
        # Buscar latest.pth
        latest = os.path.join(args.dir, 'latest.pth')
        if os.path.exists(latest):
            verifier = CheckpointVerifier(latest)
            results = verifier.verify_all_metrics()
        else:
            verify_latest_checkpoints(args.dir, 1)


if __name__ == "__main__":
    main()
