"""
Hamiltonian Metrics Computation and Tracking Module.

Implements all physics-based metrics from Hamiltonian mechanics theory:
- Hamiltonian energy conservation
- Symplectic form preservation
- Liouville measure invariance
- Phase space volume tracking
- Action integral computation
- Poisson bracket evaluation
- Spectral entropy
- Phase coherence
- Signal-to-noise ratio for reconstruction quality

Follows Single Responsibility Principle: solely responsible for metric
computation, accumulation, and retrieval.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import deque
from config import MetricsConfig


class HamiltonianMetricsTracker:
    """
    Tracks and computes all Hamiltonian mechanics metrics during
    training and inference.

    Each metric method is a pure computation with no side effects
    beyond updating internal accumulators, following the
    Interface Segregation Principle by exposing granular metric methods.
    """

    def __init__(self, config: MetricsConfig) -> None:
        self._config = config
        self._step_count: int = 0
        self._metric_history: Dict[str, deque] = {}
        self._current_metrics: Dict[str, float] = {}
        self._initialize_history_buffers()

    def _initialize_history_buffers(self) -> None:
        """Pre-allocate deque buffers for each tracked metric."""
        metric_names = [
            "hamiltonian_energy",
            "symplectic_form",
            "liouville_measure",
            "phase_space_volume",
            "action_integral",
            "poisson_bracket",
            "spectral_entropy",
            "reconstruction_snr",
            "spectral_convergence",
            "phase_coherence",
            "energy_drift",
            "gradient_norm",
            "parameter_norm",
            "learning_rate",
            "total_loss",
            "reconstruction_loss",
            "energy_conservation_loss",
            "symplectic_loss",
            "spectral_consistency_loss",
            "phase_coherence_loss",
            "action_minimization_loss",
            "liouville_loss",
            "hamiltonian_constraint_loss",
        ]
        window = self._config.moving_average_window
        for name in metric_names:
            self._metric_history[name] = deque(maxlen=window)

    def compute_hamiltonian_energy(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Hamiltonian H(q, p) = T(p) + V(q).

        T(p) = 0.5 * ||p||^2 (kinetic energy)
        V(q) = 0.5 * ||q||^2 (potential energy in harmonic approximation)

        Args:
            q: Generalized coordinates tensor (position in phase space).
            p: Conjugate momenta tensor.

        Returns:
            Scalar Hamiltonian energy value.
        """
        kinetic_energy = 0.5 * torch.sum(p ** 2)
        potential_energy = 0.5 * torch.sum(q ** 2)
        total_energy = kinetic_energy + potential_energy
        if self._config.track_hamiltonian_energy:
            self._record("hamiltonian_energy", total_energy.item())
        return total_energy

    def compute_symplectic_form(
        self, q: torch.Tensor, p: torch.Tensor, dq: torch.Tensor, dp: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the symplectic 2-form omega(dq, dp) = sum(dq_i ^ dp_i).

        Measures preservation of the canonical symplectic structure
        under Hamiltonian flow. Should remain invariant for symplectic
        integrators.

        Args:
            q: Generalized coordinates.
            p: Conjugate momenta.
            dq: Variation in coordinates.
            dp: Variation in momenta.

        Returns:
            Scalar symplectic form magnitude.
        """
        wedge_product = torch.sum(dq * dp - dp * dq)
        symplectic_norm = torch.sqrt(
            torch.sum(dq ** 2) * torch.sum(dp ** 2) + 1e-12
        )
        symplectic_form = torch.abs(wedge_product) / (symplectic_norm + 1e-12)
        if self._config.track_symplectic_form:
            self._record("symplectic_form", symplectic_form.item())
        return symplectic_form

    def compute_liouville_measure(
        self, jacobian: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Liouville measure |det(J)| for the flow map Jacobian.

        By Liouville's theorem, Hamiltonian flow preserves phase space
        volume, so det(J) should equal 1 for exact symplectic evolution.

        Args:
            jacobian: The Jacobian matrix of the phase space transformation.

        Returns:
            Absolute determinant of the Jacobian.
        """
        if jacobian.dim() == 2:
            det_j = torch.abs(torch.det(jacobian))
        else:
            batch_size = jacobian.shape[0]
            det_j = torch.mean(
                torch.abs(
                    torch.stack([torch.det(jacobian[i]) for i in range(batch_size)])
                )
            )
        if self._config.track_liouville_measure:
            self._record("liouville_measure", det_j.item())
        return det_j

    def compute_phase_space_volume(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate phase space volume occupied by the state (q, p).

        Uses the covariance ellipsoid approximation:
        V ~ sqrt(det(Cov([q, p])))

        Args:
            q: Generalized coordinates (flattened).
            p: Conjugate momenta (flattened).

        Returns:
            Estimated phase space volume.
        """
        q_flat = q.reshape(-1)
        p_flat = p.reshape(-1)
        min_len = min(q_flat.shape[0], p_flat.shape[0])
        q_flat = q_flat[:min_len]
        p_flat = p_flat[:min_len]
        phase_state = torch.stack([q_flat, p_flat], dim=0)
        covariance = torch.cov(phase_state)
        volume = torch.sqrt(torch.abs(torch.det(covariance)) + 1e-12)
        if self._config.track_phase_space_volume:
            self._record("phase_space_volume", volume.item())
        return volume

    def compute_action_integral(
        self,
        q_trajectory: torch.Tensor,
        p_trajectory: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the action integral S = integral(L dt) along a trajectory.

        L = T - V = 0.5*||p||^2 - 0.5*||q||^2 (Lagrangian)

        Args:
            q_trajectory: Sequence of coordinate states [T, ...].
            p_trajectory: Sequence of momentum states [T, ...].
            dt: Time step between trajectory points.

        Returns:
            Total action along the trajectory.
        """
        kinetic = 0.5 * torch.sum(p_trajectory ** 2, dim=list(range(1, p_trajectory.dim())))
        potential = 0.5 * torch.sum(q_trajectory ** 2, dim=list(range(1, q_trajectory.dim())))
        lagrangian = kinetic - potential
        action = torch.sum(lagrangian) * dt
        if self._config.track_action_integral:
            self._record("action_integral", action.item())
        return action

    def compute_poisson_bracket(
        self,
        f_values: torch.Tensor,
        g_values: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the Poisson bracket {f, g} = sum(df/dq * dg/dp - df/dp * dg/dq).

        Uses finite differences on the discretized phase space.

        Args:
            f_values: Observable f evaluated on phase space grid.
            g_values: Observable g evaluated on phase space grid.
            q: Coordinate grid.
            p: Momentum grid.

        Returns:
            Estimated Poisson bracket scalar.
        """
        df_dq = torch.gradient(f_values, spacing=(1.0,), dim=-1)[0]
        dg_dp = torch.gradient(g_values, spacing=(1.0,), dim=-2)[0] if g_values.dim() >= 2 else torch.zeros_like(g_values)
        df_dp = torch.gradient(f_values, spacing=(1.0,), dim=-2)[0] if f_values.dim() >= 2 else torch.zeros_like(f_values)
        dg_dq = torch.gradient(g_values, spacing=(1.0,), dim=-1)[0]
        bracket = torch.mean(df_dq * dg_dp - df_dp * dg_dq)
        if self._config.track_poisson_bracket:
            self._record("poisson_bracket", bracket.item())
        return bracket

    def compute_spectral_entropy(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral entropy H = -sum(p_i * log(p_i)).

        Measures the disorder/uniformity of the spectral distribution.
        Maximum entropy indicates uniform spectrum (white noise),
        minimum indicates pure tone (single frequency).

        Args:
            spectrum: Power spectrum tensor (non-negative).

        Returns:
            Scalar spectral entropy.
        """
        spectrum_positive = torch.abs(spectrum) + 1e-12
        normalized = spectrum_positive / torch.sum(spectrum_positive)
        entropy = -torch.sum(normalized * torch.log(normalized))
        if self._config.track_spectral_entropy:
            self._record("spectral_entropy", entropy.item())
        return entropy

    def compute_reconstruction_snr(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Signal-to-Noise Ratio in dB.

        SNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)

        Args:
            original: Ground truth signal.
            reconstructed: Reconstructed signal.

        Returns:
            SNR in decibels.
        """
        signal_power = torch.sum(original ** 2)
        noise_power = torch.sum((original - reconstructed) ** 2) + 1e-12
        snr_db = 10.0 * torch.log10(signal_power / noise_power + 1e-12)
        if self._config.track_reconstruction_snr:
            self._record("reconstruction_snr", snr_db.item())
        return snr_db

    def compute_spectral_convergence(
        self, original_spectrum: torch.Tensor, reconstructed_spectrum: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral convergence metric.

        SC = ||S_orig - S_recon||_F / ||S_orig||_F

        Lower values indicate better spectral fidelity.

        Args:
            original_spectrum: Original frequency domain representation.
            reconstructed_spectrum: Reconstructed frequency domain representation.

        Returns:
            Spectral convergence ratio.
        """
        numerator = torch.norm(original_spectrum - reconstructed_spectrum, p="fro")
        denominator = torch.norm(original_spectrum, p="fro") + 1e-12
        convergence = numerator / denominator
        if self._config.track_spectral_convergence:
            self._record("spectral_convergence", convergence.item())
        return convergence

    def compute_phase_coherence(
        self, phase_original: torch.Tensor, phase_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phase coherence between original and reconstructed signals.

        PC = |mean(exp(i * (phi_orig - phi_recon)))|

        Value of 1.0 indicates perfect phase alignment.

        Args:
            phase_original: Phase spectrum of original signal.
            phase_reconstructed: Phase spectrum of reconstructed signal.

        Returns:
            Phase coherence in [0, 1].
        """
        phase_diff = phase_original - phase_reconstructed
        coherence = torch.abs(torch.mean(torch.exp(1j * phase_diff.to(torch.complex64))))
        if self._config.track_phase_coherence:
            self._record("phase_coherence", coherence.item())
        return coherence

    def compute_energy_drift(
        self, energy_initial: float, energy_current: float
    ) -> float:
        """
        Compute relative energy drift from initial state.

        drift = |E_current - E_initial| / (|E_initial| + epsilon)

        Args:
            energy_initial: Hamiltonian energy at t=0.
            energy_current: Hamiltonian energy at current time.

        Returns:
            Relative energy drift.
        """
        drift = abs(energy_current - energy_initial) / (abs(energy_initial) + 1e-12)
        if self._config.track_energy_drift:
            self._record("energy_drift", drift)
        return drift

    def record_gradient_norm(self, model_parameters) -> float:
        """Compute and record the total gradient norm across all parameters."""
        total_norm = 0.0
        for param in model_parameters:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if self._config.track_gradient_norm:
            self._record("gradient_norm", total_norm)
        return total_norm

    def record_parameter_norm(self, model_parameters) -> float:
        """Compute and record the total parameter norm."""
        total_norm = 0.0
        for param in model_parameters:
            total_norm += param.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if self._config.track_parameter_norm:
            self._record("parameter_norm", total_norm)
        return total_norm

    def record_learning_rate(self, lr: float) -> None:
        """Record current learning rate."""
        if self._config.track_learning_rate:
            self._record("learning_rate", lr)

    def record_loss_component(self, name: str, value: float) -> None:
        """Record an individual loss component value."""
        if self._config.track_loss_components:
            self._record(name, value)

    def _record(self, metric_name: str, value: float) -> None:
        """Store a metric value in history and current snapshot."""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = deque(
                maxlen=self._config.moving_average_window
            )
        self._metric_history[metric_name].append(value)
        self._current_metrics[metric_name] = value

    def get_current_metrics(self) -> Dict[str, float]:
        """Return a snapshot of all current metric values."""
        return dict(self._current_metrics)

    def get_moving_averages(self) -> Dict[str, float]:
        """Compute moving averages for all tracked metrics."""
        averages = {}
        for name, history in self._metric_history.items():
            if len(history) > 0:
                averages[f"{name}_ma"] = sum(history) / len(history)
        return averages

    def get_formatted_metrics_string(self) -> str:
        """Format all current metrics into a human-readable string for progress bars."""
        parts = []
        for name, value in sorted(self._current_metrics.items()):
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e4:
                    parts.append(f"{name}={value:.4e}")
                else:
                    parts.append(f"{name}={value:.6f}")
            else:
                parts.append(f"{name}={value}")
        return " | ".join(parts)

    def increment_step(self) -> None:
        """Advance the global step counter."""
        self._step_count += 1

    @property
    def step_count(self) -> int:
        return self._step_count

    def should_log(self) -> bool:
        """Determine if metrics should be logged at this step."""
        return self._step_count % self._config.log_interval_steps == 0
