```text
 python precision.py --checkpoint crystal_checkpoints/crystal_epoch_5309_delta_0.3706_20260128_150931.pth
2026-01-28 17:27:35,077 - Continuation - INFO - Continuing from: crystal_checkpoints/crystal_epoch_5309_delta_0.3706_20260128_150931.pth
2026-01-28 17:27:35,087 - Continuation - INFO - Loaded checkpoint - Epoch: 5309, ValAcc: N/A, Delta: 0.37061309814453125
2026-01-28 17:27:35,709 - Continuation - INFO - Initial lambda: 7.504732e+16
2026-01-28 17:27:35,738 - Continuation - INFO - Starting epoch: 5309, delta: 0.370613
2026-01-28 17:27:39,917 - Continuation - INFO - Epoch 5359 | Loss: 2.334646e+13 | MSE: 0.003906 | Quant: 3.1109e-04 | ValAcc: 1.0000 | delta: 0.3705 -> 0.3706 | alpha: 0.99 | Sparsity: 0.00% | Cryst: 99.99% | lambda: 7.504732e+16
2026-01-28 17:27:44,331 - Continuation - INFO - Epoch 5409 | Loss: 2.328043e+13 | MSE: 0.003906 | Quant: 3.1021e-04 | ValAcc: 1.0000 | delta: 0.3704 -> 0.3706 | alpha: 0.99 | Sparsity: 0.00% | Cryst: 99.99% | lambda: 7.504732e+16
2026-01-28 17:27:44,447 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-28 17:27:44,448 - Continuation - INFO - New lambda: 7.504732e+16 -> 7.504732e+17
2026-01-28 17:27:50,490 - Continuation - INFO - Epoch 5459 | Loss: 2.322068e+14 | MSE: 0.003906 | Quant: 3.0941e-04 | ValAcc: 1.0000 | delta: 0.3703 -> 0.3706 | alpha: 0.99 | Sparsity: 0.00% | Cryst: 99.99% | lambda: 7.504732e+17
2026-01-28 17:27:56,066 - Continuation - INFO - Epoch 5509 | Loss: 2.316152e+14 | MSE: 0.003906 | Quant: 3.0863e-04 | ValAcc: 1.0000 | delta: 0.3701 -> 0.3706 | alpha: 0.99 | Sparsity: 0.00% | Cryst: 99.99% | lambda: 7.504732e+17
2026-01-28 17:27:56,293 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
...
2026-01-28 17:31:35,789 - Continuation - INFO - Epoch 7159 | Loss: 2.246780e+31 | MSE: 0.003906 | Quant: 2.9938e-04 | ValAcc: 1.0000 | delta: 0.3687 -> 0.3706 | alpha: 1.00 | Sparsity: 0.01% | Cryst: 100.00% | lambda: 7.504732e+34
2026-01-28 17:31:35,796 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 7159

---

---

❯ python3 plank.py -o results.json
DeepLeaning PLANCK - EPOCH 7196

[INPUTS]
  λ = 5.000000e-01
  δ = 0.368701
  MSE = 1.000000e+00
  Acc = 1.0000

[RESULTS]
  ħ = 2.833308e+00 [sistema]
  ħ_adimensional = 3.188592e-01
  Régimen: UNCONSTRAINED

[METHODS]
  uncertainty    : 1.359401e-01
  action         : 8.281800e+00
  conductance    : 1.000000e+00
  information    : 1.915494e+00
  Pesos: {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}

[DEREIVATED CONSTANTS]
  c_eff = 8.054496e+42 m/s
  m_P = 5.847406e+26 kg
  l_P = 6.015783e-70 m
  t_P = 7.468851e-113 s
  T_P = 2.747621e+135 K

[VS UNIVERSE]
  ħ_ratio = 2.686691e+34
  Δórdenes = +34.4
  m_P/M_☉ = 2.940656e-04

  → ħ is 34 orders more than real fisics
Saved in: results.json
❯ python verify.py crystal_checkpoints/latest.pth

======================================================================
VERIFICANDO CHECKPOINT: crystal_checkpoints/latest.pth
Epoch reportada: 7196
======================================================================

/home/grisun0/src/py/HPU-Core/verify.py:128: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at /pytorch/aten/src/ATen/native/ReduceOps.cpp:1857.)
  'std': data.std().item() if data.numel() > 0 else 0,

======================================================================
REPORTE DE VERIFICACIÓN
======================================================================

[INTEGRIDAD DE PESOS]
  Válido: True ✓

[MÉTRICAS DE VALIDACIÓN]
  MSE: 0.003906
  Accuracy: 1.0000
  Max Error: 0.245229

[MÉTRICAS DE DISCRETIZACIÓN]
  Delta (max): 0.368701
  Delta (mean): 0.126496
  Alpha: 1.00
  Is Crystal: False ✗
  Purity Index: 0.6313

  Distribución de pesos:
    Cerca de 0: 100.00%
    Cerca de 1: 0.00%
    Cerca de -1: 0.00%
    Cerca de cualquier entero: 100.00%

[MÉTRICAS DE CUANTIZACIÓN]
  Penalty: 2.993736e-04
  Total params: 589,921

[LOSS COMPLETO]
  MSE: 0.003906
  Quant penalty: 2.993736e-04
  Lambda: 7.504732e+34
  TOTAL LOSS: 2.246719e+31
  Descomposición: 0.00% MSE, 100.00% Quant

[COMPARACIÓN CON CHECKPOINT]
  delta: stored=0.368701, computed=0.368701 diff=0.00e+00 ✓
  alpha: stored=0.997770, computed=0.997770 diff=0.00e+00 ✓
  val_acc: stored=1.000000, computed=1.000000 diff=0.00e+00 ✓

[SCORE DE SALUD]
  Puntuación: 85.0/100
  Estado: HEALTHY
  Usable: True ✓
  Problemas: EXTREME_LAMBDA

======================================================================

```


