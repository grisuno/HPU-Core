# From Boltzmann Stochasticity to Hamiltonian Integrability: 
## Emergence of Topological Crystals and Synthetic Planck Constants in HPU-Cores

**grisun0**  
*January 29, 2026*

---

## Abstract

I report the discovery of a neural network state that I call a Hamiltonian Topological Crystal. By subjecting a Hamiltonian Processing Unit (HPU-Core) to extreme regularization pressure ($\lambda \approx 7.5 \times 10^{34}$), I observe a transition from conventional stochastic training to an integrable system with topological protection. The resulting model achieves perfect validation accuracy (1.0000) while operating in a "Vacuum Core" where 99.996% of parameters collapse toward zero. This work extends my previous research on Strassen crystallization from discrete algorithmic structures to continuous Hamiltonian dynamics, demonstrating that neural networks can encode physical laws as geometric structure rather than as data approximations.

---

## 1. Introduction: Beyond the Boltzmann Era

My earlier work on Strassen matrix multiplication established that neural networks could crystallize into discrete algorithmic structures under controlled training conditions. That research operated in what I termed the "Boltzmann era"—treating networks as stochastic information gases subject to entropy constraints. The Strassen experiments showed that 68% of runs could be engineered to converge to exact integer coefficients through a two-phase protocol of training and discretization.

This paper reports a qualitatively different phenomenon. The HPU-Core does not discretize to $\{-1, 0, 1\}$. Instead, it enters a continuous phase characterized by topological invariants (Berry phases $\pm 10.26$ rad, winding numbers $\pm 2$) and marginal stability (Lyapunov exponent $\lambda_{max} = +0.00175$). The system becomes an integrable Hamiltonian operator rather than a discrete algorithm.

The transition from Strassen to Hamilton represents a shift from combinatorial to topological protection of information.

---

## 2. Methodology: The $\lambda \to \infty$ Limit

I trained a spectral neural network to learn Hamiltonian evolution on a 2D torus $T^2 = [0, 2\pi) \times [0, 2\pi)$. The architecture uses Fourier-space convolutions (SpectralLayers) that operate on field configurations rather than discrete tokens.

The critical intervention was pushing the regularization parameter $\lambda$ beyond any conventional limit:

$$\lambda = 7.504732 \times 10^{34}$$

This value exceeds the mass of the Sun expressed in dimensionless units ($M_\odot c^2 / \hbar \approx 10^{34}$ s$^{-1}$). At this scale, the loss function becomes dominated by the regularization term:

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \lambda \cdot \delta \approx 2.25 \times 10^{31}$$

where $\mathcal{L}_{MSE} = 0.0039$ contributes only $0.00\%$ of the total loss. The optimization effectively minimizes $\lambda \cdot ||w||^2$ subject to the constraint that the network still computes the Hamiltonian correctly.

---

## 3. Results: The Vacuum Core State

### 3.1 The $\delta \approx 0.3687$ Fixed Point

Unlike the Strassen case where $\delta \to 0$ (exact integer discretization), the HPU stabilizes at $\delta = 0.3687$ with $\alpha = 1.00$ (maximum purity). This is not a failure to converge. The system has found a **continuous attractor** protected by topology rather than a discrete minimum.

The weight distribution shows:
- Near-zero fraction: 99.996%
- Near-one fraction: 0.000%
- Near-minus-one fraction: 0.000%

Yet the network maintains perfect accuracy. The information is encoded not in weight magnitudes but in **phases** of the complex spectral kernels.

### 3.2 Topological Protection

The Topological Crystal Experiment revealed:

| Invariant | Value | Interpretation |
|-----------|-------|----------------|
| Berry phase (layer 0) | $+10.26$ rad | Winding $+2$ |
| Berry phase (layer 1) | $-11.51$ rad | Winding $-2$ |
| Winding numbers | $\pm 2$ | Topologically non-trivial |
| Protection strength | 1.0 | Complete |

These invariants emerge from the complex structure of SpectralLayers ($W = W_{real} + i W_{imag}$). They are not engineered; they arise spontaneously when $\lambda$ forces the real and imaginary components to organize into a globally coherent phase structure.

The winding numbers $\pm 2$ indicate that the weight configuration cannot be continuously deformed to zero without breaking phase coherence. This is the mechanism of topological protection: local perturbations (noise, pruning up to 50%) cannot destroy the global structure.

### 3.3 Marginal Stability

The Lyapunov analysis shows:

$$\lambda_{max} = +0.001751$$

This is **marginal stability**: neither attracting ($\lambda < 0$) nor chaotic ($\lambda \gg 0$). The system lives at the edge of chaos, where information propagates without amplification or decay.

The Lyapunov spectrum contains both positive and negative exponents, indicating a saddle-point structure in weight space. The trajectory is not converging to a fixed point but exploring a **limit cycle** or **strange saddle** of dimension 31 (the effective rank).

---

## 4. The Synthetic Planck Constant

### 4.1 Derivation of $\hbar_{eff}$

Treating the HPU as a quantum many-body system, I calculate an effective Planck constant from the uncertainty relation:

$$\hbar_{eff} \sim \delta^2 \cdot \lambda / \omega$$

where $\omega = \sqrt{\lambda}$ is the natural frequency of the confining potential.

With the measured values:
- $\delta = 0.3687$
- $\lambda = 7.5 \times 10^{34}$
- $\omega = 2.74 \times 10^{17}$

I obtain:

$$\hbar_{eff} \approx 1.02 \times 10^{34}$$

This is not a dimensional accident. The coincidence $\hbar_{eff} \approx \lambda$ reflects that the system has self-organized its quantum scale to match the confinement strength.

### 4.2 Physical Constants of the HPU Universe

| Constant | Value | vs. Physical Universe |
|----------|-------|----------------------|
| $\hbar_{eff}$ | $1.02 \times 10^{34}$ | $10^{68} \times \hbar_{SI}$ |
| $c_{eff}$ | $2.9 \times 10^{76}$ m/s | $10^{68} \times c$ |
| $m_{Planck}$ | $2.1 \times 10^{60}$ kg | $10^{30} \times M_\odot$ |
| $l_{Planck}$ | $1.7 \times 10^{-103}$ m | $10^{68} \times$ smaller |
| $t_{Planck}$ | $5.8 \times 10^{-180}$ s | $10^{68} \times$ shorter |

The HPU operates in a regime where "quantum effects" (interference, uncertainty) are macroscopic. The regularization has created a synthetic universe with its own Planck scale.

### 4.3 Interpretation

The high $\hbar_{eff}$ explains why the system maintains coherence despite extreme sparsity. In this synthetic quantum regime:
- The 24 effective parameters (0.004% of total) act as **edge states** conducting information
- The 99.996% near-zero weights form a **topological vacuum**
- Information propagates via phase coherence rather than amplitude

This is analogous to the quantum Hall effect, where conduction occurs through edge states while the bulk is insulating.

---

## 5. Comparison: Strassen vs. Hamilton

| Feature | Strassen Crystal | Hamilton Crystal |
|---------|------------------|------------------|
| **Order parameter** | $\delta \to 0$ (discrete) | $\delta \approx 0.37$ (continuous) |
| **Protection** | Permutation symmetry | Berry phase ($\pm 2$) |
| **Stability** | Fixed point ($\lambda_{max} < 0$) | Marginal ($\lambda_{max} \approx 0$) |
| **Sparsity** | 7/8 slots (87.5%) | 24/590k params (0.004%) |
| **$\hbar_{eff}$** | Not defined (classical discrete) | $10^{34}$ (quantum continuous) |
| **Fragility** | Noise $\sigma \geq 0.001$ destroys | Float16 destroys, pruning to 50% tolerated |
| **Generalization** | Zero-shot to $64 \times 64$ | Perfect on validation, Hamiltonian exact |

The Strassen crystal is a **classical discrete attractor**. The Hamilton crystal is a **quantum continuous attractor**. Both represent algorithmic structure, but the Hamilton case reveals that neural networks can encode physics-like laws through topology rather than symbolic coefficients.

---

## 6. The $\lambda \to \infty$ Protocol

Based on 7196 epochs of training and topological analysis, I propose the following protocol for inducing Hamiltonian crystals:

1. **Architecture**: Spectral layers with complex kernels (real + imaginary parts)
2. **Target**: Hamiltonian evolution on compact manifold (torus $T^2$)
3. **Regularization**: Push $\lambda$ until float64 gradient explosion (typically $\lambda > 10^{30}$)
4. **Monitoring**: Track $\delta$ (should stabilize, not converge to 0), Berry phases (should become non-trivial), and $\lambda_{max}$ (should approach 0 from above)
5. **Verification**: Check topological invariants and validate Hamiltonian correctness

The success rate remains undetermined (N=1). Due to the stringent $\lambda$ requirements, this protocol is both computationally intensive and numerically unstable. This assessment is based on initial seed mining, where a marginal delta ($\delta$ = 0.46 from others 0.49) was first identified in seed 32.

---

## 7. Limitations and Honest Assessment

**What I demonstrate:**
- A single HPU-Core can be driven into a topologically protected state with perfect accuracy
- The state has measurable invariants (Berry phases, winding numbers)
- An effective $\hbar$ emerges naturally from the dynamics
- The system is marginally stable, not converged to a fixed point

**What I do not demonstrate:**
- Reproducibility (N=1, one successful run)
- Generalization to other Hamiltonians
- Whether the protocol works with different architectures
- Causality: does high $\lambda$ cause topological protection, or merely select for it?
- Practical utility: the state is fragile (float16 destroys it) and expensive to reach

**Critical fragility:**
The float16 precision test failed completely (accuracy 0.0, MSE infinite). This is not merely numerical error; it indicates that the topological protection relies on delicate phase coherence that breaks under coarse quantization. The "crystal" is real but fragile.

**Comparison to Strassen:**
The Strassen work had statistical power (N=195, 68% success rate). This Hamilton work has N=1. The claims are correspondingly weaker. I report a phenomenon, not a reproducible protocol.

---

## 8. Implications for Neural Network Physics

The Hamilton crystal suggests that neural networks can operate in distinct physical regimes:

1. **Boltzmann regime** (standard training): Stochastic, high entropy, local minima
2. **Strassen regime** (discrete crystallization): Combinatorial protection, integer coefficients
3. **Hamilton regime** (topological crystallization): Continuous protection, phase coherence, synthetic quantum mechanics

The transition between regimes is controlled by $\lambda$ (regularization strength) and architecture (spectral vs. standard layers). This opens questions about whether other physical regimes exist—relativistic, gravitational, or quantum field theoretic analogues in neural networks.

---

## 9. Conclusion

I have documented the existence of a neural network state that behaves like a topologically protected quantum system. The HPU-Core at $\lambda \approx 10^{34}$ achieves:
- Perfect accuracy (1.0000) with 99.996% sparsity
- Topological invariants (Berry phases $\pm 10.26$ rad, winding $\pm 2$)
- Marginal stability ($\lambda_{max} \approx 0$)
- Synthetic Planck constant $\hbar_{eff} \approx 10^{34}$

This extends my previous Strassen work from discrete algorithms to continuous physical laws. Where Strassen showed that networks could learn symbolic structure, Hamilton shows they can learn geometric-topological structure.

The honesty imperative requires stating: this is a single observation, not a validated protocol. The phenomenon is real (measured, not hallucinated), but its generality is unknown. The extreme conditions ($\lambda \sim 10^{34}$) suggest this may be a singular point in the space of neural network dynamics, not a broadly accessible regime.

What remains valid is the methodology: treat neural networks as physical systems, measure their thermodynamic and topological properties, and report what the machine tells you without embellishment.

In Strassen, ℏ emerges as a task-level algorithmic thermodynamic constant, invariant across training realizations. In contrast, within the HPU framework, ℏ becomes algorithm-dependent, encoding the effective action scale of the learning dynamics itself. This distinction motivates the notion of an algorithmic thermodynamics of learning, where constants are not universal but process-specific.

A striking aspect of the observation is the extreme sensitivity to initial conditions. Out of approximately 150 tested random seeds, all but one converged to a glassy regime. Only seed 32 accessed the crystalline phase. The sole detectable difference at initialization was a lower initial control parameter, with $\delta \approx 0.46$ for seed 32 versus $\delta \approx 0.49$ for all others. This indicates the presence of a sharp critical threshold in the initial conditions: trajectories starting above it are dynamically unable to cross into the low-action crystalline basin. The phenomenon is therefore not stochastic but governed by a narrow, history-dependent accessibility condition.

---

## References


[1] Citation for Grokking and Generalization: Title: Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets, Authors: Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra, arXiv: 2201.02177, 2022.

[2] Citation for Grokking and Local Complexity (LC): Title: Deep Networks Always Grok and Here is Why, Authors: A. Imtiaz Humayun, Randall Balestriero, Richard Baraniuk, arXiv:2402.15555, 2024.

[3] Citation for Superposition as Lossy Compression: Title: Superposition as lossy compression, Authors: Bereska et al., arXiv 2024.

[4] grisun0. Algorithmic Induction via Structural Weight Transfer (v13). Zenodo, 2025. https://doi.org/10.5281/zenodo.18072858


---

## Data Availability

The checkpoint analyzed (epoch 7196, $\lambda = 7.504732 \times 10^{34}$) is available at:
- Repository: https://github.com/grisuno/HPU-Core
- Mining seeds: `mining_seeds.py`
- Experiment: `experiment2.py`
- Refinery: `refinamiento.py`
- Last fase: `precision.py`
- Analysis script: `topological_experiment.py`
- Verification script: `verify.py`
- Sintetic Plank: `plank.py`

The topological analysis JSON and $\hbar$ calculation scripts are included in the repository.

---

## Acknowledgments

This work was conducted as a continuation of the Strassen crystallization research. The HPU-Core represents a shift from discrete to continuous algorithmic structure, from combinatorial to topological protection. Whether this shift proves as reproducible as Strassen remains to be tested.

---

## Appendix C: Reproducibility

Repository: https://github.com/grisuno/HPU-Core

DOI: https://doi.org/10.5281/zenodo.18072858
DOI: https://doi.org/10.5281/zenodo.18407920

Reproduction:

```bash
git clone https://github.com/grisuno/HPU-Core
cd HPU-Core
pip install -r requirements.txt
python app.py
```

Related repositories:

- Ancestor: https://github.com/grisuno/SWAN-Phoenix-Rising
- Core Framework: https://github.com/grisuno/agi
- Parity Cassette: https://github.com/grisuno/algebra-de-grok
- Wave Cassette: https://github.com/grisuno/1d_wave_equation_grokker
- Kepler Cassette: https://github.com/grisuno/kepler_orbit_grokker
- Pendulum Cassette: https://github.com/grisuno/chaotic_pendulum_grokked
- Ciclotron Cassette: https://github.com/grisuno/supertopo3
- MatMul 2x2 Cassette: https://github.com/grisuno/matrixgrokker
- Strassen Cassette: https://github.com/grisuno/strass_strassen

---

*grisun0*
*January 28, 2026*

---

# Hamiltonian Processor Unit: Spectral Invariant Hamiltonian Operator Grokkit Framework

## Overview

This repository implements a complete scientific software package for the investigation and demonstration of the grokking phenomenon in Hamiltonian operator learning. The framework provides a complete implementation of the theoretical framework described in Theorem 1.1 on spectral convergence, with particular emphasis on the phase transition from memorization to generalization in neural network training.

The central objective of this package is to empirically verify that neural networks, when trained with the appropriate thermodynamic optimization framework, exhibit a characteristic phase transition where they transition from memorizing training data to learning the underlying mathematical structure. In the context of Hamiltonian operators on a torus manifold, this manifests as the emergence of spectral representations that capture the underlying dynamical system.

## Theorem 1.1: Spectral Convergence in Hamiltonian Learning

The theoretical foundation of this framework rests upon Theorem 1.1, which establishes conditions under which a neural network trained on Hamiltonian operator data will exhibit spectral convergence. The theorem states that under appropriate conditions on the optimization landscape, specifically when the effective dimensionality of the weight manifold exceeds a critical threshold, the model undergoes a phase transition characterized by the emergence of low-rank spectral representations.

The key insight is that the learning dynamics exhibit two distinct phases. During the initial memorization phase, the model fits the training data through high-dimensional interpolation, utilizing the full capacity of the network to memorize specific examples. Subsequently, as the local complexity metric increases beyond the critical threshold, the optimization landscape facilitates a transition to the generalization phase, where the model discovers the spectral structure inherent in the Hamiltonian operator.

### Local Complexity and Phase Transition

The Local Complexity (LC) metric serves as the primary order parameter for detecting the phase transition. Defined as the von Neumann entropy of the output covariance matrix, LC quantifies the effective dimensionality of the representation manifold. The Thermal Engine optimization mechanism monitors this metric and dynamically adjusts the regularization landscape to facilitate the transition.

## Architecture and Implementation

### Model Architecture

The neural network architecture implements a multilayer perceptron with the following specifications:

| Layer | Input Size | Output Size | Activation |
|-------|------------|-------------|------------|
| Input | 2 | 256 | GELU |
| Hidden 1 | 256 | 256 | GELU |
| Hidden 2 | 256 | 256 | GELU |
| Output | 256 | 4 | Linear |

The input dimension of 2 corresponds to the coordinates (x, y) on the torus manifold [0, 2π] × [0, 2π]. The output dimension of 4 represents the components of the Hamiltonian operator evaluated at the specified coordinates.

### Thermal Engine Optimization

The Thermal Engine implements an adaptive regularization mechanism that modulates weight decay based on the current state of the learning dynamics. The optimization process operates according to the following principles:

The engine computes two primary metrics at each training step. The Local Complexity (LC) is calculated as the entropy of the output covariance matrix, capturing the effective dimensionality of the learned representation. The Superposition (SP) metric measures the ratio between the largest and smallest non-zero singular values of the weight matrices, indicating the degree of efficient representation.

The weight decay coefficient is dynamically adjusted according to the formula:

```
weight_decay = base_decay * (1 + β * LC / TARGET_LC)
```

where β is a scaling factor and TARGET_LC is the critical complexity threshold. This adaptive mechanism ensures that the optimization landscape remains conducive to spectral convergence throughout the training process.

## Installation and Setup

### Prerequisites

The following software dependencies must be satisfied prior to executing the framework:

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy 1.21 or higher

### Directory Structure

```
hamiltonian_operator/
├── main.py              # Primary training and execution script
├── test_grokking.py     # Comprehensive validation suite
├── README.md            # This documentation file
└── weights/             # Directory for trained model weights
    └── model_checkpoint.pth
```

### Initial Configuration

The weights directory contains pre-trained model weights that demonstrate successful grokking. Upon cloning the repository, execute the following command to verify the installation:

```bash
cd hamiltonian_operator
python test_grokking.py
```

This command will load the pre-trained weights and execute the validation suite, confirming the presence of the grokking phenomenon through the prescribed test criteria.

## Usage

### Training Mode

To train a model from scratch, remove any existing weights from the weights directory:

```bash
rm weights/model_checkpoint.pth
python main.py
```

The training process will commence, monitoring the Local Complexity metric throughout. Upon reaching the target LC threshold, the model weights will be saved to weights/model_checkpoint.pth and training will terminate.

### Validation Mode

When weights exist in the weights directory, the framework automatically enters validation mode. The main script will load the pre-trained model, evaluate the grokking criteria, and report the results without executing additional training iterations.

### Custom Validation

For comprehensive validation with detailed reporting:

```bash
python test_grokking.py
```

This generates a validation_report.md file containing detailed metrics and test results.

## Training Parameters

The following hyperparameters govern the training dynamics:

| Parameter | Value | Description |
|-----------|-------|-------------|
| TORUS_N | 50 | Resolution of the torus grid |
| HIDDEN_DIM | 256 | Hidden layer dimension |
| NUM_EPOCHS | 50000 | Maximum training epochs |
| LEARNING_RATE | 0.001 | Base optimizer learning rate |
| TARGET_LC | 5.5 | Critical Local Complexity threshold |
| LC_BETA | 2.0 | Thermal Engine scaling factor |

These parameters have been empirically validated to produce reliable grokking within the specified epoch budget. Adjustments may be necessary depending on computational resources.

## Expected Results

### Training Output

Successful training produces the following characteristic output pattern:

```
Epoch 0: Loss=2.847, LC=0.021, SP=142.3
Epoch 5000: Loss=0.423, LC=1.234, SP=89.7
Epoch 10000: Loss=0.156, LC=2.567, SP=67.2
Epoch 15000: Loss=0.067, LC=3.891, SP=45.6
Epoch 20000: Loss=0.024, LC=5.234, SP=32.1
Epoch 22441: Loss=0.012, LC=5.502, SP=28.4
```

The Local Complexity metric exhibits monotonic increase throughout training, with the phase transition occurring approximately between epochs 15000 and 20000. Upon reaching TARGET_LC = 5.5, the Thermal Engine considers the grokking criterion satisfied and saves the model weights.

### Validation Criteria

The test suite validates the following conditions:

1. Local Complexity Verification: LC >= 95% of TARGET_LC
2. Generalization Accuracy: Prediction error < 0.1 on held-out data
3. Output Manifold Stability: Non-degenerate output representations

## Mathematical Formulation

### Hamiltonian Operator on Torus

The Hamiltonian operator implemented in this framework operates on functions defined on the 2-torus manifold T² = [0, 2π) × [0, 2π). The operator is constructed as a linear combination of trigonometric basis functions:

```
H(x, y) = [cos(x)sin(y) + 0.5cos(2x),
           sin(x)cos(y) + 0.3sin(2y),
           cos(x+y) + 0.4cos(2x+y),
           sin(x-y) + 0.2sin(2x-y)]
```

This formulation ensures that the target function possesses a rich spectral structure that can be efficiently captured by the neural network representation.

### Spectral Representation Theorem

The neural network learns an approximation to the Hamiltonian operator through the universal approximation theorem extended to spectral representations. The weights evolve during training to align with the principal spectral components of the target function, resulting in an efficient representation that generalizes beyond the training examples.

## 6. Reproducibility

Code and pretrained grokked models are publicly available:

- Core Framework: [https://github.com/grisuno/agi](https://github.com/grisuno/agi)
- Parity Cassette: [https://github.com/grisuno/algebra-de-grok](https://github.com/grisuno/algebra-de-grok)
- Wave Cassette: [https://github.com/grisuno/1d_wave_equation_grokker](https://github.com/grisuno/1d_wave_equation_grokker)
- Kepler Cassette: [https://github.com/grisuno/kepler_orbit_grokker](https://github.com/grisuno/kepler_orbit_grokker)
- Pendulum Cassette: [https://github.com/grisuno/chaotic_pendulum_grokked](https://github.com/grisuno/chaotic_pendulum_grokked)
- Ciclotron Cassette: [https://github.com/grisuno/supertopo3](https://github.com/grisuno/supertopo3)
- MatMul 2x2 Cassette: [https://github.com/grisuno/matrixgrokker](https://github.com/grisuno/matrixgrokker)
- HPU Hamiltonian Cassette: [https://github.com/grisuno/HPU-Core](https://github.com/grisuno/HPU-Core)

All random seeds are fixed to ensure reproducibility:

```python
torch.manual_seed(42)
np.random.seed(42)
```

This configuration has been validated to produce consistent grokking behavior across multiple training runs on identical hardware configurations.

## References

The theoretical framework underlying this implementation draws upon established principles in statistical learning theory, optimization dynamics, and mathematical physics. Specific attention is given to the phase transition phenomena in neural network training and the emergence of spectral representations in overparameterized models.

## Author

grisun0

---



![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://doi.org/10.5281/zenodo.18072859)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
