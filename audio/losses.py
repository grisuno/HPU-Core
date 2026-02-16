"""
Hamiltonian Loss Functions Module.

Implements the complete loss function from the Hamiltonian mechanics framework:
- Reconstruction loss (MSE)
- Energy conservation loss
- Symplectic structure preservation loss
- Spectral consistency loss
- Phase coherence loss
- Action minimization loss (principle of least action)
- Liouville theorem loss (phase space volume conservation)
- Hamiltonian constraint loss (Hamilton's equations satisfaction)

Follows Single Responsibility Principle: each loss term is an independent
computation that can be composed.

Follows Open/Closed Principle: new loss terms can be added without
modifying existing computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
from config import TrainingConfig


class HamiltonianLossComputer:
    """
    Computes the composite Hamiltonian loss function with all
    physics-based regularization terms.

    Each loss component is independently weighted via TrainingConfig,
    enabling fine-grained control over the training objective.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._mse_loss = nn.MSELoss()

    def compute_total_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        intermediates: List[torch.Tensor],
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the complete weighted loss with all Hamiltonian terms.

        Args:
            prediction: Model output [B, 1, H, W].
            target: Ground truth [B, 1, H, W].
            intermediates: List of intermediate hidden states from forward pass.
            model: The model (for parameter access in regularization).

        Returns:
            Tuple of (total_loss tensor, dict of individual loss values).
        """
        loss_components: Dict[str, float] = {}
        reconstruction = self._compute_reconstruction_loss(prediction, target)
        loss_components["reconstruction_loss"] = reconstruction.item()
        energy_conservation = self._compute_energy_conservation_loss(intermediates)
        loss_components["energy_conservation_loss"] = energy_conservation.item()
        symplectic = self._compute_symplectic_loss(intermediates)
        loss_components["symplectic_loss"] = symplectic.item()
        spectral_consistency = self._compute_spectral_consistency_loss(
            prediction, target
        )
        loss_components["spectral_consistency_loss"] = spectral_consistency.item()
        phase_coherence = self._compute_phase_coherence_loss(prediction, target)
        loss_components["phase_coherence_loss"] = phase_coherence.item()
        action_minimization = self._compute_action_minimization_loss(intermediates)
        loss_components["action_minimization_loss"] = action_minimization.item()
        liouville = self._compute_liouville_loss(intermediates)
        loss_components["liouville_loss"] = liouville.item()
        hamiltonian_constraint = self._compute_hamiltonian_constraint_loss(
            intermediates
        )
        loss_components["hamiltonian_constraint_loss"] = hamiltonian_constraint.item()
        total_loss = (
            self._config.loss_reconstruction_weight * reconstruction
            + self._config.loss_energy_conservation_weight * energy_conservation
            + self._config.loss_symplectic_weight * symplectic
            + self._config.loss_spectral_consistency_weight * spectral_consistency
            + self._config.loss_phase_coherence_weight * phase_coherence
            + self._config.loss_action_minimization_weight * action_minimization
            + self._config.loss_liouville_weight * liouville
            + self._config.loss_hamiltonian_constraint_weight * hamiltonian_constraint
        )
        loss_components["total_loss"] = total_loss.item()
        return total_loss, loss_components

    def _compute_reconstruction_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """MSE reconstruction loss between predicted and target spectrograms."""
        return self._mse_loss(prediction, target)

    def _compute_energy_conservation_loss(
        self, intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Penalize energy drift across layers.

        The Hamiltonian energy E = 0.5 * ||phi||^2 should remain
        approximately constant through the evolution layers.
        """
        if len(intermediates) < 2:
            return torch.tensor(0.0, requires_grad=True)
        energies = []
        for state in intermediates:
            energy = 0.5 * torch.mean(state ** 2)
            energies.append(energy)
        energy_drifts = []
        initial_energy = energies[0]
        for energy in energies[1:]:
            drift = (energy - initial_energy) ** 2 / (initial_energy ** 2 + 1e-12)
            energy_drifts.append(drift)
        return torch.mean(torch.stack(energy_drifts))

    def _compute_symplectic_loss(
        self, intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Penalize violation of symplectic structure.

        For pairs of consecutive states (q_i, q_{i+1}), we interpret
        q as position and dq = q_{i+1} - q_i as a proxy for momentum.
        The symplectic form dq ^ dp should be preserved.
        """
        if len(intermediates) < 3:
            return torch.tensor(0.0, requires_grad=True)
        symplectic_violations = []
        for i in range(len(intermediates) - 2):
            q = intermediates[i]
            dq_1 = intermediates[i + 1] - intermediates[i]
            dq_2 = intermediates[i + 2] - intermediates[i + 1]
            area_1 = torch.sum(q * dq_1, dim=(1, 2, 3))
            area_2 = torch.sum(intermediates[i + 1] * dq_2, dim=(1, 2, 3))
            violation = torch.mean((area_1 - area_2) ** 2)
            symplectic_violations.append(violation)
        return torch.mean(torch.stack(symplectic_violations))

    def _compute_spectral_consistency_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize spectral divergence in frequency domain.

        ||FFT(prediction) - FFT(target)||_F / ||FFT(target)||_F
        """
        pred_fft = torch.fft.rfft2(prediction)
        target_fft = torch.fft.rfft2(target)
        numerator = torch.norm(
            torch.abs(pred_fft) - torch.abs(target_fft), p=2
        )
        denominator = torch.norm(torch.abs(target_fft), p=2) + 1e-12
        return numerator / denominator

    def _compute_phase_coherence_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize phase misalignment between prediction and target.

        1 - |mean(exp(i * (angle(FFT(pred)) - angle(FFT(target)))))|
        """
        pred_fft = torch.fft.rfft2(prediction)
        target_fft = torch.fft.rfft2(target)
        phase_diff = torch.angle(pred_fft) - torch.angle(target_fft)
        coherence = torch.abs(
            torch.mean(torch.exp(1j * phase_diff.to(torch.complex64)))
        )
        return 1.0 - coherence

    def _compute_action_minimization_loss(
        self, intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Principle of least action: minimize the total action
        S = sum(|phi_{i+1} - phi_i|) along the trajectory.
        """
        if len(intermediates) < 2:
            return torch.tensor(0.0, requires_grad=True)
        actions = []
        for i in range(len(intermediates) - 1):
            action_density = torch.abs(intermediates[i + 1] - intermediates[i])
            actions.append(torch.mean(action_density))
        return torch.mean(torch.stack(actions))

    def _compute_liouville_loss(
        self, intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Liouville theorem: phase space volume should be preserved.

        We approximate this by checking that the variance of hidden
        states remains approximately constant through evolution.
        """
        if len(intermediates) < 2:
            return torch.tensor(0.0, requires_grad=True)
        variances = []
        for state in intermediates:
            var = torch.var(state)
            variances.append(var)
        var_drifts = []
        initial_var = variances[0]
        for var in variances[1:]:
            drift = (var - initial_var) ** 2 / (initial_var ** 2 + 1e-12)
            var_drifts.append(drift)
        return torch.mean(torch.stack(var_drifts))

    def _compute_hamiltonian_constraint_loss(
        self, intermediates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.

        Approximated by checking time-reversal symmetry:
        the forward evolution followed by reverse should return
        to the initial state.
        """
        if len(intermediates) < 3:
            return torch.tensor(0.0, requires_grad=True)
        forward_trajectory = intermediates
        mid_idx = len(forward_trajectory) // 2
        forward_delta = forward_trajectory[mid_idx] - forward_trajectory[0]
        backward_delta = forward_trajectory[-1] - forward_trajectory[mid_idx]
        symmetry_violation = torch.mean(
            (torch.norm(forward_delta, dim=(1, 2, 3)) - torch.norm(backward_delta, dim=(1, 2, 3))) ** 2
        )
        return symmetry_violation
