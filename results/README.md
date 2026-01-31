# Results

```text
❯ python3 refinamiento.py --checkpoint checkpoints/latest.pth
2026-01-31 15:03:37,725 - Crystallization - INFO - Cargando checkpoint: checkpoints/latest.pth
2026-01-31 15:03:37,735 - Crystallization - INFO - Checkpoint cargado - Época: 4975, ValAcc: 1.0000, Delta: 0.4591
2026-01-31 15:03:38,368 - Crystallization - INFO - Optimizador creado: SGD lr=1e-05, wd=0.01
2026-01-31 15:03:38,396 - Crystallization - INFO - ============================================================
2026-01-31 15:03:38,396 - Crystallization - INFO - INICIANDO CRISTALIZACIÓN FORZADA
2026-01-31 15:03:38,396 - Crystallization - INFO - Objetivo: δ < 0.05
2026-01-31 15:03:38,396 - Crystallization - INFO - LR: 1e-05, WD: 0.01
2026-01-31 15:03:38,396 - Crystallization - INFO - Lambda quant inicial: 0.5
2026-01-31 15:03:38,396 - Crystallization - INFO - ============================================================
2026-01-31 15:03:42,327 - Crystallization - INFO - Época 5025 | Loss: 0.043572 | MSE: 0.003906 | Quant: 0.0793 | ValAcc: 1.0000 | δ: 0.4590 → 0.4591 | α: 0.78 | Sparsity: 0.00% | Cryst: 68.53% | λ: 0.500
2026-01-31 15:03:46,209 - Crystallization - INFO - Época 5075 | Loss: 0.043562 | MSE: 0.003906 | Quant: 0.0793 | ValAcc: 1.0000 | δ: 0.4589 → 0.4591 | α: 0.78 | Sparsity: 0.00% | Cryst: 68.54% | λ: 0.500
2026-01-31 15:03:46,316 - Crystallization - INFO - Sin mejora en 101 épocas. Aumentando agresividad...
2026-01-31 15:03:46,317 - Crystallization - INFO - Nueva λ: 0.500 → 1.500, LR reducido en 10%
2026-01-31 15:03:51,307 - Crystallization - INFO - Época 5125 | Loss: 0.121431 | MSE: 0.003906 | Quant: 0.0784 | ValAcc: 1.0000 | δ: 0.4578 → 0.4591 | α: 0.78 | Sparsity: 0.00% | Cryst: 69.15% | λ: 1.500
2026-01-31 15:03:56,421 - Crystallization - INFO - Época 5175 | Loss: 0.119967 | MSE: 0.003906 | Quant: 0.0774 | ValAcc: 1.0000 | δ: 0.4645 → 0.4591 | α: 0.77 | Sparsity: 0.00% | Cryst: 69.75% | λ: 1.500
2026-01-31 15:03:56,621 - Crystallization - INFO - Sin mejora en 202 épocas. Aumentando agresividad...
2026-01-31 15:03:56,622 - Crystallization - INFO - Nueva λ: 1.500 → 4.500, LR reducido en 10%
2026-01-31 15:04:01,666 - Crystallization - INFO - Época 5225 | Loss: 0.347725 | MSE: 0.003906 | Quant: 0.0764 | ValAcc: 1.0000 | δ: 0.4711 → 0.4591 | α: 0.75 | Sparsity: 0.00% | Cryst: 70.37% | λ: 4.500
2026-01-31 15:04:07,232 - Crystallization - INFO - Época 5275 | Loss: 0.343413 | MSE: 0.003906 | Quant: 0.0754 | ValAcc: 1.0000 | δ: 0.4777 → 0.4591 | α: 0.74 | Sparsity: 0.00% | Cryst: 70.98% | λ: 4.500
2026-01-31 15:04:07,530 - Crystallization - INFO - Sin mejora en 303 épocas. Aumentando agresividad...
2026-01-31 15:04:07,531 - Crystallization - INFO - Nueva λ: 4.500 → 13.500, LR reducido en 10%
2026-01-31 15:04:12,215 - Crystallization - INFO - Época 5325 | Loss: 1.009417 | MSE: 0.003906 | Quant: 0.0745 | ValAcc: 1.0000 | δ: 0.4842 → 0.4591 | α: 0.73 | Sparsity: 0.00% | Cryst: 71.60% | λ: 13.500
2026-01-31 15:04:17,168 - Crystallization - INFO - Época 5375 | Loss: 0.996547 | MSE: 0.003906 | Quant: 0.0735 | ValAcc: 1.0000 | δ: 0.4906 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 72.23% | λ: 13.500
2026-01-31 15:04:17,567 - Crystallization - INFO - Sin mejora en 404 épocas. Aumentando agresividad...
2026-01-31 15:04:17,568 - Crystallization - INFO - Nueva λ: 13.500 → 40.500, LR reducido en 10%
2026-01-31 15:04:22,268 - Crystallization - INFO - Época 5425 | Loss: 2.941690 | MSE: 0.003906 | Quant: 0.0725 | ValAcc: 1.0000 | δ: 0.4968 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 72.88% | λ: 40.500
2026-01-31 15:04:27,118 - Crystallization - INFO - Época 500: Podados 32 parámetros
2026-01-31 15:04:27,214 - Crystallization - INFO - Época 5475 | Loss: 2.901759 | MSE: 0.003906 | Quant: 0.0716 | ValAcc: 1.0000 | δ: 0.4969 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 73.49% | λ: 40.500
2026-01-31 15:04:27,707 - Crystallization - INFO - Sin mejora en 505 épocas. Aumentando agresividad...
2026-01-31 15:04:27,708 - Crystallization - INFO - Nueva λ: 40.500 → 121.500, LR reducido en 10%
2026-01-31 15:04:32,175 - Crystallization - INFO - Época 5525 | Loss: 8.561516 | MSE: 0.003906 | Quant: 0.0704 | ValAcc: 1.0000 | δ: 0.4969 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 74.17% | λ: 121.500
2026-01-31 15:04:37,051 - Crystallization - INFO - Época 5575 | Loss: 8.424309 | MSE: 0.003906 | Quant: 0.0693 | ValAcc: 1.0000 | δ: 0.4981 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 74.89% | λ: 121.500
2026-01-31 15:04:37,653 - Crystallization - INFO - Sin mejora en 606 épocas. Aumentando agresividad...
2026-01-31 15:04:37,653 - Crystallization - INFO - Nueva λ: 121.500 → 364.500, LR reducido en 10%
2026-01-31 15:04:42,013 - Crystallization - INFO - Época 5625 | Loss: 24.704979 | MSE: 0.003906 | Quant: 0.0678 | ValAcc: 1.0000 | δ: 0.4957 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 75.78% | λ: 364.500
2026-01-31 15:04:46,993 - Crystallization - INFO - Época 5675 | Loss: 24.122828 | MSE: 0.003906 | Quant: 0.0662 | ValAcc: 1.0000 | δ: 0.4933 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 76.72% | λ: 364.500
2026-01-31 15:04:47,693 - Crystallization - INFO - Sin mejora en 707 épocas. Aumentando agresividad...
2026-01-31 15:04:47,693 - Crystallization - INFO - Nueva λ: 364.500 → 1093.500, LR reducido en 10%
2026-01-31 15:04:51,916 - Crystallization - INFO - Época 5725 | Loss: 69.908147 | MSE: 0.003906 | Quant: 0.0639 | ValAcc: 1.0000 | δ: 0.4981 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 78.00% | λ: 1093.500
2026-01-31 15:04:57,278 - Crystallization - INFO - Época 5775 | Loss: 67.343335 | MSE: 0.003906 | Quant: 0.0616 | ValAcc: 1.0000 | δ: 0.4988 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 79.19% | λ: 1093.500
2026-01-31 15:04:58,049 - Crystallization - INFO - Sin mejora en 808 épocas. Aumentando agresividad...
2026-01-31 15:04:58,049 - Crystallization - INFO - Nueva λ: 1093.500 → 3280.500, LR reducido en 10%
2026-01-31 15:05:02,229 - Crystallization - INFO - Época 5825 | Loss: 194.506415 | MSE: 0.003906 | Quant: 0.0593 | ValAcc: 1.0000 | δ: 0.4942 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 80.42% | λ: 3280.500
2026-01-31 15:05:07,241 - Crystallization - INFO - Época 5875 | Loss: 187.167834 | MSE: 0.003906 | Quant: 0.0571 | ValAcc: 1.0000 | δ: 0.4919 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 81.62% | λ: 3280.500
2026-01-31 15:05:08,186 - Crystallization - INFO - Sin mejora en 909 épocas. Aumentando agresividad...
2026-01-31 15:05:08,186 - Crystallization - INFO - Nueva λ: 3280.500 → 9841.500, LR reducido en 10%
2026-01-31 15:05:12,423 - Crystallization - INFO - Época 5925 | Loss: 540.007593 | MSE: 0.003906 | Quant: 0.0549 | ValAcc: 1.0000 | δ: 0.4966 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 82.76% | λ: 9841.500
2026-01-31 15:05:17,278 - Crystallization - INFO - Época 1000: Podados 39 parámetros
2026-01-31 15:05:17,378 - Crystallization - INFO - Época 5975 | Loss: 519.038049 | MSE: 0.003906 | Quant: 0.0527 | ValAcc: 1.0000 | δ: 0.4980 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 83.87% | λ: 9841.500
2026-01-31 15:05:18,396 - Crystallization - INFO - Sin mejora en 1010 épocas. Aumentando agresividad...
2026-01-31 15:05:18,397 - Crystallization - INFO - Nueva λ: 9841.500 → 29524.500, LR reducido en 10%
2026-01-31 15:05:22,345 - Crystallization - INFO - Época 6025 | Loss: 1495.775464 | MSE: 0.003906 | Quant: 0.0507 | ValAcc: 1.0000 | δ: 0.4982 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 84.96% | λ: 29524.500
^[[B2026-01-31 15:05:27,292 - Crystallization - INFO - Época 6075 | Loss: 1435.981519 | MSE: 0.003906 | Quant: 0.0486 | ValAcc: 1.0000 | δ: 0.4953 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 85.98% | λ: 29524.500
2026-01-31 15:05:28,481 - Crystallization - INFO - Sin mejora en 1111 épocas. Aumentando agresividad...
2026-01-31 15:05:28,482 - Crystallization - INFO - Nueva λ: 29524.500 → 88573.500, LR reducido en 10%
2026-01-31 15:05:32,862 - Crystallization - INFO - Época 6125 | Loss: 4133.157910 | MSE: 0.003906 | Quant: 0.0467 | ValAcc: 1.0000 | δ: 0.4973 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 86.98% | λ: 88573.500
2026-01-31 15:05:37,848 - Crystallization - INFO - Época 6175 | Loss: 3962.935693 | MSE: 0.003906 | Quant: 0.0447 | ValAcc: 1.0000 | δ: 0.4987 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 87.91% | λ: 88573.500
2026-01-31 15:05:39,054 - Crystallization - INFO - Sin mejora en 1212 épocas. Aumentando agresividad...
2026-01-31 15:05:39,054 - Crystallization - INFO - Nueva λ: 88573.500 → 265720.500, LR reducido en 10%
2026-01-31 15:05:42,893 - Crystallization - INFO - Época 6225 | Loss: 11391.777344 | MSE: 0.003906 | Quant: 0.0429 | ValAcc: 1.0000 | δ: 0.4909 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 88.83% | λ: 265720.500
2026-01-31 15:05:47,824 - Crystallization - INFO - Época 6275 | Loss: 10908.125000 | MSE: 0.003906 | Quant: 0.0411 | ValAcc: 1.0000 | δ: 0.4832 → 0.4591 | α: 0.73 | Sparsity: 0.00% | Cryst: 89.67% | λ: 265720.500
2026-01-31 15:05:49,135 - Crystallization - INFO - Sin mejora en 1313 épocas. Aumentando agresividad...
2026-01-31 15:05:49,136 - Crystallization - INFO - Nueva λ: 265720.500 → 797161.500, LR reducido en 10%
2026-01-31 15:05:52,842 - Crystallization - INFO - Época 6325 | Loss: 31313.384375 | MSE: 0.003906 | Quant: 0.0393 | ValAcc: 1.0000 | δ: 0.4755 → 0.4591 | α: 0.74 | Sparsity: 0.00% | Cryst: 90.47% | λ: 797161.500
2026-01-31 15:05:57,961 - Crystallization - INFO - Época 6375 | Loss: 29942.144922 | MSE: 0.003906 | Quant: 0.0376 | ValAcc: 1.0000 | δ: 0.4757 → 0.4591 | α: 0.74 | Sparsity: 0.00% | Cryst: 91.22% | λ: 797161.500
2026-01-31 15:05:59,457 - Crystallization - INFO - Sin mejora en 1414 épocas. Aumentando agresividad...
2026-01-31 15:05:59,457 - Crystallization - INFO - Nueva λ: 797161.500 → 2391484.500, LR reducido en 10%
2026-01-31 15:06:03,070 - Crystallization - INFO - Época 6425 | Loss: 85829.839063 | MSE: 0.003906 | Quant: 0.0359 | ValAcc: 1.0000 | δ: 0.4806 → 0.4591 | α: 0.73 | Sparsity: 0.00% | Cryst: 91.94% | λ: 2391484.500
^[[B2026-01-31 15:06:08,252 - Crystallization - INFO - Época 1500: Podados 49 parámetros
2026-01-31 15:06:08,365 - Crystallization - INFO - Época 6475 | Loss: 81940.167188 | MSE: 0.003906 | Quant: 0.0343 | ValAcc: 1.0000 | δ: 0.4854 → 0.4591 | α: 0.72 | Sparsity: 0.00% | Cryst: 92.65% | λ: 2391484.500
^[[B2026-01-31 15:06:09,888 - Crystallization - INFO - Sin mejora en 1515 épocas. Aumentando agresividad...
2026-01-31 15:06:09,888 - Crystallization - INFO - Nueva λ: 2391484.500 → 7174453.500, LR reducido en 10%
2026-01-31 15:06:13,450 - Crystallization - INFO - Época 6525 | Loss: 234530.343750 | MSE: 0.003906 | Quant: 0.0327 | ValAcc: 1.0000 | δ: 0.4902 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 93.30% | λ: 7174453.500
2026-01-31 15:06:18,878 - Crystallization - INFO - Época 6575 | Loss: 223583.884375 | MSE: 0.003906 | Quant: 0.0312 | ValAcc: 1.0000 | δ: 0.4949 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 93.90% | λ: 7174453.500
2026-01-31 15:06:20,523 - Crystallization - INFO - Sin mejora en 1616 épocas. Aumentando agresividad...
2026-01-31 15:06:20,523 - Crystallization - INFO - Nueva λ: 7174453.500 → 21523360.500, LR reducido en 10%
2026-01-31 15:06:23,857 - Crystallization - INFO - Época 6625 | Loss: 638927.162500 | MSE: 0.003906 | Quant: 0.0297 | ValAcc: 1.0000 | δ: 0.4996 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 94.46% | λ: 21523360.500
2026-01-31 15:06:29,015 - Crystallization - INFO - Época 6675 | Loss: 608105.037500 | MSE: 0.003906 | Quant: 0.0283 | ValAcc: 1.0000 | δ: 0.4995 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 94.99% | λ: 21523360.500
2026-01-31 15:06:30,765 - Crystallization - INFO - Sin mejora en 1717 épocas. Aumentando agresividad...
2026-01-31 15:06:30,766 - Crystallization - INFO - Nueva λ: 21523360.500 → 64570081.500, LR reducido en 10%
2026-01-31 15:06:34,015 - Crystallization - INFO - Época 6725 | Loss: 1734818.225000 | MSE: 0.003906 | Quant: 0.0269 | ValAcc: 1.0000 | δ: 0.4976 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 95.47% | λ: 64570081.500
2026-01-31 15:06:38,978 - Crystallization - INFO - Época 6775 | Loss: 1648273.125000 | MSE: 0.003906 | Quant: 0.0255 | ValAcc: 1.0000 | δ: 0.4971 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 95.92% | λ: 64570081.500
2026-01-31 15:06:40,717 - Crystallization - INFO - Sin mejora en 1818 épocas. Aumentando agresividad...
2026-01-31 15:06:40,717 - Crystallization - INFO - Nueva λ: 64570081.500 → 193710244.500, LR reducido en 10%
2026-01-31 15:06:44,169 - Crystallization - INFO - Época 6825 | Loss: 4693945.900000 | MSE: 0.003906 | Quant: 0.0242 | ValAcc: 1.0000 | δ: 0.4971 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 96.36% | λ: 193710244.500
2026-01-31 15:06:49,233 - Crystallization - INFO - Época 6875 | Loss: 4451699.500000 | MSE: 0.003906 | Quant: 0.0230 | ValAcc: 1.0000 | δ: 0.4927 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 96.74% | λ: 193710244.500
2026-01-31 15:06:51,334 - Crystallization - INFO - Sin mejora en 1919 épocas. Aumentando agresividad...
2026-01-31 15:06:51,334 - Crystallization - INFO - Nueva λ: 193710244.500 → 581130733.500, LR reducido en 10%
2026-01-31 15:06:54,424 - Crystallization - INFO - Época 6925 | Loss: 12653862.400000 | MSE: 0.003906 | Quant: 0.0218 | ValAcc: 1.0000 | δ: 0.4974 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 97.09% | λ: 581130733.500
2026-01-31 15:06:59,238 - Crystallization - INFO - Época 2000: Podados 52 parámetros
2026-01-31 15:06:59,338 - Crystallization - INFO - Época 6975 | Loss: 11976788.200000 | MSE: 0.003906 | Quant: 0.0206 | ValAcc: 1.0000 | δ: 0.4983 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 97.42% | λ: 581130733.500
2026-01-31 15:07:01,375 - Crystallization - INFO - Sin mejora en 2020 épocas. Aumentando agresividad...
2026-01-31 15:07:01,375 - Crystallization - INFO - Nueva λ: 581130733.500 → 1743392200.500, LR reducido en 10%
2026-01-31 15:07:04,323 - Crystallization - INFO - Época 7025 | Loss: 33976736.000000 | MSE: 0.003906 | Quant: 0.0195 | ValAcc: 1.0000 | δ: 0.4999 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 97.71% | λ: 1743392200.500
2026-01-31 15:07:09,238 - Crystallization - INFO - Época 7075 | Loss: 32095257.600000 | MSE: 0.003906 | Quant: 0.0184 | ValAcc: 1.0000 | δ: 0.4969 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 97.97% | λ: 1743392200.500
2026-01-31 15:07:11,318 - Crystallization - INFO - Sin mejora en 2121 épocas. Aumentando agresividad...
2026-01-31 15:07:11,318 - Crystallization - INFO - Nueva λ: 1743392200.500 → 5230176601.500, LR reducido en 10%
2026-01-31 15:07:14,160 - Crystallization - INFO - Época 7125 | Loss: 90858332.800000 | MSE: 0.003906 | Quant: 0.0174 | ValAcc: 1.0000 | δ: 0.4992 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 98.22% | λ: 5230176601.500
2026-01-31 15:07:19,056 - Crystallization - INFO - Época 7175 | Loss: 85642784.000000 | MSE: 0.003906 | Quant: 0.0164 | ValAcc: 1.0000 | δ: 0.4938 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 98.44% | λ: 5230176601.500
2026-01-31 15:07:21,213 - Crystallization - INFO - Sin mejora en 2222 épocas. Aumentando agresividad...
2026-01-31 15:07:21,213 - Crystallization - INFO - Nueva λ: 5230176601.500 → 15690529804.500, LR reducido en 10%
2026-01-31 15:07:23,960 - Crystallization - INFO - Época 7225 | Loss: 241909424.000000 | MSE: 0.003906 | Quant: 0.0154 | ValAcc: 1.0000 | δ: 0.4860 → 0.4591 | α: 0.72 | Sparsity: 0.00% | Cryst: 98.63% | λ: 15690529804.500
2026-01-31 15:07:28,853 - Crystallization - INFO - Época 7275 | Loss: 227514835.200000 | MSE: 0.003906 | Quant: 0.0145 | ValAcc: 1.0000 | δ: 0.4880 → 0.4591 | α: 0.72 | Sparsity: 0.00% | Cryst: 98.81% | λ: 15690529804.500
2026-01-31 15:07:31,112 - Crystallization - INFO - Sin mejora en 2323 épocas. Aumentando agresividad...
2026-01-31 15:07:31,112 - Crystallization - INFO - Nueva λ: 15690529804.500 → 47071589413.500, LR reducido en 10%
2026-01-31 15:07:33,779 - Crystallization - INFO - Época 7325 | Loss: 641182067.200000 | MSE: 0.003906 | Quant: 0.0136 | ValAcc: 1.0000 | δ: 0.4928 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 98.96% | λ: 47071589413.500
2026-01-31 15:07:38,615 - Crystallization - INFO - Época 7375 | Loss: 601586726.400000 | MSE: 0.003906 | Quant: 0.0128 | ValAcc: 1.0000 | δ: 0.4975 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.10% | λ: 47071589413.500
2026-01-31 15:07:40,942 - Crystallization - INFO - Sin mejora en 2424 épocas. Aumentando agresividad...
2026-01-31 15:07:40,942 - Crystallization - INFO - Nueva λ: 47071589413.500 → 141214768240.500, LR reducido en 10%
2026-01-31 15:07:43,785 - Crystallization - INFO - Época 7425 | Loss: 1691170483.200000 | MSE: 0.003906 | Quant: 0.0120 | ValAcc: 1.0000 | δ: 0.4966 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.23% | λ: 141214768240.500
2026-01-31 15:07:49,776 - Crystallization - INFO - Época 7475 | Loss: 1582697190.400000 | MSE: 0.003906 | Quant: 0.0112 | ValAcc: 1.0000 | δ: 0.4924 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 99.35% | λ: 141214768240.500
2026-01-31 15:07:52,734 - Crystallization - INFO - Sin mejora en 2525 épocas. Aumentando agresividad...
2026-01-31 15:07:52,735 - Crystallization - INFO - Nueva λ: 141214768240.500 → 423644304721.500, LR reducido en 10%
2026-01-31 15:07:55,830 - Crystallization - INFO - Época 7525 | Loss: 4437844172.800000 | MSE: 0.003906 | Quant: 0.0105 | ValAcc: 1.0000 | δ: 0.4971 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.44% | λ: 423644304721.500
2026-01-31 15:08:00,844 - Crystallization - INFO - Época 7575 | Loss: 4142357145.600000 | MSE: 0.003906 | Quant: 0.0098 | ValAcc: 1.0000 | δ: 0.4971 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.53% | λ: 423644304721.500
2026-01-31 15:08:03,534 - Crystallization - INFO - Sin mejora en 2626 épocas. Aumentando agresividad...
2026-01-31 15:08:03,534 - Crystallization - INFO - Nueva λ: 423644304721.500 → 1270932914164.500, LR reducido en 10%
2026-01-31 15:08:06,221 - Crystallization - INFO - Época 7625 | Loss: 11583583846.400000 | MSE: 0.003906 | Quant: 0.0091 | ValAcc: 1.0000 | δ: 0.4964 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.59% | λ: 1270932914164.500
2026-01-31 15:08:11,827 - Crystallization - INFO - Época 7675 | Loss: 10782136320.000000 | MSE: 0.003906 | Quant: 0.0085 | ValAcc: 1.0000 | δ: 0.4998 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 99.66% | λ: 1270932914164.500
2026-01-31 15:08:14,747 - Crystallization - INFO - Sin mejora en 2727 épocas. Aumentando agresividad...
2026-01-31 15:08:14,747 - Crystallization - INFO - Nueva λ: 1270932914164.500 → 3812798742493.500, LR reducido en 10%
2026-01-31 15:08:17,029 - Crystallization - INFO - Época 7725 | Loss: 30064931225.599998 | MSE: 0.003906 | Quant: 0.0079 | ValAcc: 1.0000 | δ: 0.4927 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 99.71% | λ: 3812798742493.500
2026-01-31 15:08:21,973 - Crystallization - INFO - Época 7775 | Loss: 27903288934.400002 | MSE: 0.003906 | Quant: 0.0073 | ValAcc: 1.0000 | δ: 0.4919 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 99.76% | λ: 3812798742493.500
2026-01-31 15:08:24,805 - Crystallization - INFO - Sin mejora en 2828 épocas. Aumentando agresividad...
2026-01-31 15:08:24,806 - Crystallization - INFO - Nueva λ: 3812798742493.500 → 11438396227480.500, LR reducido en 10%
2026-01-31 15:08:26,999 - Crystallization - INFO - Época 7825 | Loss: 77576423014.399994 | MSE: 0.003906 | Quant: 0.0068 | ValAcc: 1.0000 | δ: 0.4966 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.80% | λ: 11438396227480.500
2026-01-31 15:08:31,883 - Crystallization - INFO - Época 7875 | Loss: 71773700096.000000 | MSE: 0.003906 | Quant: 0.0063 | ValAcc: 1.0000 | δ: 0.4980 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.84% | λ: 11438396227480.500
2026-01-31 15:08:34,901 - Crystallization - INFO - Sin mejora en 2929 épocas. Aumentando agresividad...
2026-01-31 15:08:34,901 - Crystallization - INFO - Nueva λ: 11438396227480.500 → 34315188682441.500, LR reducido en 10%
2026-01-31 15:08:37,263 - Crystallization - INFO - Época 7925 | Loss: 198885179392.000000 | MSE: 0.003906 | Quant: 0.0058 | ValAcc: 1.0000 | δ: 0.4990 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.87% | λ: 34315188682441.500
2026-01-31 15:08:42,487 - Crystallization - INFO - Época 7975 | Loss: 183396201267.200012 | MSE: 0.003906 | Quant: 0.0053 | ValAcc: 1.0000 | δ: 0.4978 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.89% | λ: 34315188682441.500
2026-01-31 15:08:45,194 - Crystallization - INFO - Sin mejora en 3030 épocas. Aumentando agresividad...
2026-01-31 15:08:45,194 - Crystallization - INFO - Nueva λ: 34315188682441.500 → 102945566047324.500, LR reducido en 10%
2026-01-31 15:08:47,221 - Crystallization - INFO - Época 8025 | Loss: 506516340736.000000 | MSE: 0.003906 | Quant: 0.0049 | ValAcc: 1.0000 | δ: 0.4960 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.91% | λ: 102945566047324.500
2026-01-31 15:08:51,768 - Crystallization - INFO - Época 8075 | Loss: 465431848550.400024 | MSE: 0.003906 | Quant: 0.0045 | ValAcc: 1.0000 | δ: 0.4953 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.93% | λ: 102945566047324.500
2026-01-31 15:08:54,489 - Crystallization - INFO - Sin mejora en 3131 épocas. Aumentando agresividad...
2026-01-31 15:08:54,489 - Crystallization - INFO - Nueva λ: 102945566047324.500 → 308836698141973.500, LR reducido en 10%
2026-01-31 15:08:56,173 - Crystallization - INFO - Época 8125 | Loss: 1280671953715.199951 | MSE: 0.003906 | Quant: 0.0041 | ValAcc: 1.0000 | δ: 0.5000 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 99.95% | λ: 308836698141973.500
2026-01-31 15:09:00,545 - Crystallization - INFO - Época 8175 | Loss: 1172444518809.600098 | MSE: 0.003906 | Quant: 0.0038 | ValAcc: 1.0000 | δ: 0.4964 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.96% | λ: 308836698141973.500
2026-01-31 15:09:03,301 - Crystallization - INFO - Sin mejora en 3232 épocas. Aumentando agresividad...
2026-01-31 15:09:03,302 - Crystallization - INFO - Nueva λ: 308836698141973.500 → 926510094425920.500, LR reducido en 10%
2026-01-31 15:09:04,869 - Crystallization - INFO - Época 8225 | Loss: 3213963506483.200195 | MSE: 0.003906 | Quant: 0.0035 | ValAcc: 1.0000 | δ: 0.4911 → 0.4591 | α: 0.71 | Sparsity: 0.00% | Cryst: 99.96% | λ: 926510094425920.500
2026-01-31 15:09:09,207 - Crystallization - INFO - Época 8275 | Loss: 2931044646912.000000 | MSE: 0.003906 | Quant: 0.0032 | ValAcc: 1.0000 | δ: 0.4958 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.97% | λ: 926510094425920.500
2026-01-31 15:09:12,073 - Crystallization - INFO - Sin mejora en 3333 épocas. Aumentando agresividad...
2026-01-31 15:09:12,074 - Crystallization - INFO - Nueva λ: 926510094425920.500 → 2779530283277761.500, LR reducido en 10%
2026-01-31 15:09:13,556 - Crystallization - INFO - Época 8325 | Loss: 8003505461657.599609 | MSE: 0.003906 | Quant: 0.0029 | ValAcc: 1.0000 | δ: 0.4993 → 0.4591 | α: 0.69 | Sparsity: 0.00% | Cryst: 99.98% | λ: 2779530283277761.500
2026-01-31 15:09:17,877 - Crystallization - INFO - Época 8375 | Loss: 7269864754380.799805 | MSE: 0.003906 | Quant: 0.0026 | ValAcc: 1.0000 | δ: 0.4968 → 0.4591 | α: 0.70 | Sparsity: 0.00% | Cryst: 99.98% | λ: 2779530283277761.500
2026-01-31 15:09:20,823 - Crystallization - INFO - Sin mejora en 3434 épocas. Aumentando agresividad...
2026-01-31 15:09:20,824 - Crystallization - INFO - Nueva λ: 2779530283277761.500 → 8338590849833284.000, LR reducido en 10%
2026-01-31 15:09:22,203 - Crystallization - INFO - Época 8425 | Loss: 19770924806963.199219 | MSE: 0.003906 | Quant: 0.0024 | ValAcc: 1.0000 | δ: 0.4890 → 0.4591 | α: 0.72 | Sparsity: 0.00% | Cryst: 99.98% | λ: 8338590849833284.000
2026-01-31 15:09:26,581 - Crystallization - INFO - Época 8475 | Loss: 17885103233433.601562 | MSE: 0.003906 | Quant: 0.0021 | ValAcc: 1.0000 | δ: 0.4813 → 0.4591 | α: 0.73 | Sparsity: 0.00% | Cryst: 99.99% | λ: 8338590849833284.000
2026-01-31 15:09:29,596 - Crystallization - INFO - Sin mejora en 3535 épocas. Aumentando agresividad...
2026-01-31 15:09:29,596 - Crystallization - INFO - Nueva λ: 8338590849833284.000 → 25015772549499852.000, LR reducido en 10%
2026-01-31 15:09:30,912 - Crystallization - INFO - Época 8525 | Loss: 48438974192025.601562 | MSE: 0.003906 | Quant: 0.0019 | ValAcc: 1.0000 | δ: 0.4737 → 0.4591 | α: 0.75 | Sparsity: 0.00% | Cryst: 99.99% | λ: 25015772549499852.000
2026-01-31 15:09:35,412 - Crystallization - INFO - Época 8575 | Loss: 43628282825932.796875 | MSE: 0.003906 | Quant: 0.0017 | ValAcc: 1.0000 | δ: 0.4662 → 0.4591 | α: 0.76 | Sparsity: 0.00% | Cryst: 99.99% | λ: 25015772549499852.000
2026-01-31 15:09:38,526 - Crystallization - INFO - Sin mejora en 3636 épocas. Aumentando agresividad...
2026-01-31 15:09:38,526 - Crystallization - INFO - Nueva λ: 25015772549499852.000 → 75047317648499552.000, LR reducido en 10%
2026-01-31 15:09:39,726 - Crystallization - INFO - Época 8625 | Loss: 117603173820006.406250 | MSE: 0.003906 | Quant: 0.0016 | ValAcc: 1.0000 | δ: 0.4588 → 0.4591 | α: 0.78 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:09:42,222 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8654_delta_0.4545_20260131_150942.pth
2026-01-31 15:09:42,222 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.454522
2026-01-31 15:09:44,139 - Crystallization - INFO - Época 8675 | Loss: 105409451288166.406250 | MSE: 0.003906 | Quant: 0.0014 | ValAcc: 1.0000 | δ: 0.4515 → 0.4545 | α: 0.80 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:09:45,110 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8686_delta_0.4499_20260131_150945.pth
2026-01-31 15:09:45,111 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.449861
2026-01-31 15:09:47,965 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8718_delta_0.4452_20260131_150947.pth
2026-01-31 15:09:47,966 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.445237
2026-01-31 15:09:48,598 - Crystallization - INFO - Época 8725 | Loss: 94244764332851.203125 | MSE: 0.003906 | Quant: 0.0013 | ValAcc: 1.0000 | δ: 0.4442 → 0.4452 | α: 0.81 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:09:50,855 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8750_delta_0.4406_20260131_150950.pth
2026-01-31 15:09:50,855 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.440650
2026-01-31 15:09:53,109 - Crystallization - INFO - Época 8775 | Loss: 84072067183411.203125 | MSE: 0.003906 | Quant: 0.0011 | ValAcc: 1.0000 | δ: 0.4371 → 0.4406 | α: 0.83 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:09:53,649 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8781_delta_0.4362_20260131_150953.pth
2026-01-31 15:09:53,649 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.436241
2026-01-31 15:09:56,461 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8812_delta_0.4319_20260131_150956.pth
2026-01-31 15:09:56,461 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.431865
2026-01-31 15:09:57,647 - Crystallization - INFO - Época 8825 | Loss: 74824000995328.000000 | MSE: 0.003906 | Quant: 0.0010 | ValAcc: 1.0000 | δ: 0.4300 → 0.4319 | α: 0.84 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:09:59,226 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8843_delta_0.4275_20260131_150959.pth
2026-01-31 15:09:59,226 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.427524
2026-01-31 15:10:02,233 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8874_delta_0.4232_20260131_151002.pth
2026-01-31 15:10:02,233 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.423216
2026-01-31 15:10:02,323 - Crystallization - INFO - Época 8875 | Loss: 66424783083929.601562 | MSE: 0.003906 | Quant: 0.0009 | ValAcc: 1.0000 | δ: 0.4231 → 0.4232 | α: 0.86 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:05,009 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8905_delta_0.4189_20260131_151005.pth
2026-01-31 15:10:05,009 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.418942
2026-01-31 15:10:06,816 - Crystallization - INFO - Época 8925 | Loss: 58821144949555.203125 | MSE: 0.003906 | Quant: 0.0008 | ValAcc: 1.0000 | δ: 0.4162 → 0.4189 | α: 0.88 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:07,788 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8936_delta_0.4147_20260131_151007.pth
2026-01-31 15:10:07,788 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.414700
2026-01-31 15:10:10,628 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8967_delta_0.4105_20260131_151010.pth
2026-01-31 15:10:10,628 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.410492
2026-01-31 15:10:11,387 - Crystallization - INFO - Época 8975 | Loss: 51954698525081.601562 | MSE: 0.003906 | Quant: 0.0007 | ValAcc: 1.0000 | δ: 0.4094 → 0.4105 | α: 0.89 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:13,452 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_8998_delta_0.4063_20260131_151013.pth
2026-01-31 15:10:13,453 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.406315
2026-01-31 15:10:15,878 - Crystallization - INFO - Época 9025 | Loss: 45766433308672.000000 | MSE: 0.003906 | Quant: 0.0006 | ValAcc: 1.0000 | δ: 0.4027 → 0.4063 | α: 0.91 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:16,246 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9029_delta_0.4022_20260131_151016.pth
2026-01-31 15:10:16,246 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.402172
2026-01-31 15:10:19,061 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9060_delta_0.3981_20260131_151019.pth
2026-01-31 15:10:19,062 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.398060
2026-01-31 15:10:20,433 - Crystallization - INFO - Época 9075 | Loss: 40205176274944.000000 | MSE: 0.003906 | Quant: 0.0005 | ValAcc: 1.0000 | δ: 0.3961 → 0.3981 | α: 0.93 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:21,851 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9091_delta_0.3940_20260131_151021.pth
2026-01-31 15:10:21,852 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.393979
2026-01-31 15:10:24,669 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9122_delta_0.3899_20260131_151024.pth
2026-01-31 15:10:24,670 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.389931
2026-01-31 15:10:24,942 - Crystallization - INFO - Época 9125 | Loss: 35239777166950.398438 | MSE: 0.003906 | Quant: 0.0005 | ValAcc: 1.0000 | δ: 0.3895 → 0.3899 | α: 0.94 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:27,461 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9153_delta_0.3859_20260131_151027.pth
2026-01-31 15:10:27,461 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.385913
2026-01-31 15:10:29,477 - Crystallization - INFO - Época 9175 | Loss: 30803851476992.000000 | MSE: 0.003906 | Quant: 0.0004 | ValAcc: 1.0000 | δ: 0.3831 → 0.3859 | α: 0.96 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:30,299 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9184_delta_0.3819_20260131_151030.pth
2026-01-31 15:10:30,300 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.381927
2026-01-31 15:10:33,021 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9214_delta_0.3781_20260131_151033.pth
2026-01-31 15:10:33,021 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.378099
2026-01-31 15:10:34,043 - Crystallization - INFO - Época 9225 | Loss: 26855916699648.000000 | MSE: 0.003906 | Quant: 0.0004 | ValAcc: 1.0000 | δ: 0.3767 → 0.3781 | α: 0.98 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:35,894 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9244_delta_0.3743_20260131_151035.pth
2026-01-31 15:10:35,894 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.374299
2026-01-31 15:10:38,743 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9274_delta_0.3705_20260131_151038.pth
2026-01-31 15:10:38,743 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.370527
2026-01-31 15:10:38,836 - Crystallization - INFO - Época 9275 | Loss: 23350764109824.000000 | MSE: 0.003906 | Quant: 0.0003 | ValAcc: 1.0000 | δ: 0.3704 → 0.3705 | α: 0.99 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:41,533 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9304_delta_0.3668_20260131_151041.pth
2026-01-31 15:10:41,533 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.366784
2026-01-31 15:10:43,428 - Crystallization - INFO - Época 9325 | Loss: 20255339655987.199219 | MSE: 0.003906 | Quant: 0.0003 | ValAcc: 1.0000 | δ: 0.3642 → 0.3668 | α: 1.01 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:44,229 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9334_delta_0.3631_20260131_151044.pth
2026-01-31 15:10:44,229 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.363069
2026-01-31 15:10:46,970 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9364_delta_0.3594_20260131_151046.pth
2026-01-31 15:10:46,970 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.359381
2026-01-31 15:10:47,969 - Crystallization - INFO - Época 9375 | Loss: 17522438334054.400391 | MSE: 0.003906 | Quant: 0.0002 | ValAcc: 1.0000 | δ: 0.3580 → 0.3594 | α: 1.03 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:49,613 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9394_delta_0.3557_20260131_151049.pth
2026-01-31 15:10:49,614 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.355721
2026-01-31 15:10:52,262 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9424_delta_0.3521_20260131_151052.pth
2026-01-31 15:10:52,263 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.352089
2026-01-31 15:10:52,356 - Crystallization - INFO - Época 9425 | Loss: 15122102327705.599609 | MSE: 0.003906 | Quant: 0.0002 | ValAcc: 1.0000 | δ: 0.3520 → 0.3521 | α: 1.04 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:54,899 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9454_delta_0.3485_20260131_151054.pth
2026-01-31 15:10:54,899 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.348484
2026-01-31 15:10:56,799 - Crystallization - INFO - Época 9475 | Loss: 13020393635840.000000 | MSE: 0.003906 | Quant: 0.0002 | ValAcc: 1.0000 | δ: 0.3460 → 0.3485 | α: 1.06 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:10:57,597 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9484_delta_0.3449_20260131_151057.pth
2026-01-31 15:10:57,597 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.344905
2026-01-31 15:11:00,326 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9514_delta_0.3414_20260131_151100.pth
2026-01-31 15:11:00,326 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.341353
2026-01-31 15:11:01,348 - Crystallization - INFO - Época 9525 | Loss: 11180039366246.400391 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3401 → 0.3414 | α: 1.08 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:11:03,027 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9544_delta_0.3378_20260131_151103.pth
2026-01-31 15:11:03,027 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.337828
2026-01-31 15:11:05,663 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9573_delta_0.3344_20260131_151105.pth
2026-01-31 15:11:05,663 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.334446
2026-01-31 15:11:05,857 - Crystallization - INFO - Época 9575 | Loss: 9575712214220.800781 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3342 → 0.3344 | α: 1.10 | Sparsity: 0.00% | Cryst: 99.99% | λ: 75047317648499552.000
2026-01-31 15:11:08,281 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9602_delta_0.3311_20260131_151108.pth
2026-01-31 15:11:08,281 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.331087
2026-01-31 15:11:10,367 - Crystallization - INFO - Época 9625 | Loss: 8183570707251.200195 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3284 → 0.3311 | α: 1.11 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:10,918 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9631_delta_0.3278_20260131_151110.pth
2026-01-31 15:11:10,919 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.327754
2026-01-31 15:11:13,483 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9660_delta_0.3244_20260131_151113.pth
2026-01-31 15:11:13,484 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.324444
2026-01-31 15:11:15,064 - Crystallization - INFO - Época 9675 | Loss: 6983670405529.599609 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3227 → 0.3244 | α: 1.13 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:16,329 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9689_delta_0.3212_20260131_151116.pth
2026-01-31 15:11:16,329 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.321158
2026-01-31 15:11:18,930 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9718_delta_0.3179_20260131_151118.pth
2026-01-31 15:11:18,930 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.317896
2026-01-31 15:11:19,579 - Crystallization - INFO - Época 9725 | Loss: 5954301172121.599609 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3171 → 0.3179 | α: 1.15 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:21,580 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9747_delta_0.3147_20260131_151121.pth
2026-01-31 15:11:21,580 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.314657
2026-01-31 15:11:24,157 - Crystallization - INFO - Época 9775 | Loss: 5077656508825.599609 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3116 → 0.3147 | α: 1.17 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:24,258 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9776_delta_0.3114_20260131_151124.pth
2026-01-31 15:11:24,258 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.311442
2026-01-31 15:11:27,115 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9805_delta_0.3083_20260131_151127.pth
2026-01-31 15:11:27,115 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.308250
2026-01-31 15:11:28,902 - Crystallization - INFO - Época 9825 | Loss: 4332285539123.200195 | MSE: 0.003906 | Quant: 0.0001 | ValAcc: 1.0000 | δ: 0.3061 → 0.3083 | α: 1.18 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:29,708 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9834_delta_0.3051_20260131_151129.pth
2026-01-31 15:11:29,708 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.305082
2026-01-31 15:11:32,814 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9863_delta_0.3019_20260131_151132.pth
2026-01-31 15:11:32,814 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.301936
2026-01-31 15:11:34,187 - Crystallization - INFO - Época 9875 | Loss: 3696565367603.200195 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.3006 → 0.3019 | α: 1.20 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:35,802 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9892_delta_0.2988_20260131_151135.pth
2026-01-31 15:11:35,802 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.298813
2026-01-31 15:11:38,334 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9920_delta_0.2958_20260131_151138.pth
2026-01-31 15:11:38,334 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.295818
2026-01-31 15:11:38,767 - Crystallization - INFO - Época 9925 | Loss: 3152116344422.399902 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2953 → 0.2958 | α: 1.22 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:40,783 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9948_delta_0.2928_20260131_151140.pth
2026-01-31 15:11:40,783 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.292845
2026-01-31 15:11:43,268 - Crystallization - INFO - Época 9975 | Loss: 2693331419136.000000 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2900 → 0.2928 | α: 1.24 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:43,365 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_9976_delta_0.2899_20260131_151143.pth
2026-01-31 15:11:43,365 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.289893
2026-01-31 15:11:45,999 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10004_delta_0.2870_20260131_151145.pth
2026-01-31 15:11:45,999 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.286961
2026-01-31 15:11:47,911 - Crystallization - INFO - Época 10025 | Loss: 2306602749132.799805 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2848 → 0.2870 | α: 1.26 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:48,547 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10032_delta_0.2840_20260131_151148.pth
2026-01-31 15:11:48,547 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.284050
2026-01-31 15:11:51,082 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10060_delta_0.2812_20260131_151151.pth
2026-01-31 15:11:51,082 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.281158
2026-01-31 15:11:52,451 - Crystallization - INFO - Época 10075 | Loss: 1983142245171.199951 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2796 → 0.2812 | α: 1.27 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:53,609 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10088_delta_0.2783_20260131_151153.pth
2026-01-31 15:11:53,609 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.278288
2026-01-31 15:11:56,168 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10116_delta_0.2754_20260131_151156.pth
2026-01-31 15:11:56,168 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.275437
2026-01-31 15:11:57,026 - Crystallization - INFO - Época 10125 | Loss: 1713510285312.000000 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2745 → 0.2754 | α: 1.29 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:11:58,736 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10144_delta_0.2726_20260131_151158.pth
2026-01-31 15:11:58,736 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.272606
2026-01-31 15:12:01,325 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10172_delta_0.2698_20260131_151201.pth
2026-01-31 15:12:01,325 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.269795
2026-01-31 15:12:01,600 - Crystallization - INFO - Época 10175 | Loss: 1492083251609.600098 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2695 → 0.2698 | α: 1.31 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:03,854 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10200_delta_0.2670_20260131_151203.pth
2026-01-31 15:12:03,854 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.267003
2026-01-31 15:12:06,132 - Crystallization - INFO - Época 10225 | Loss: 1309442729574.399902 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2645 → 0.2670 | α: 1.33 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:06,324 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10227_delta_0.2643_20260131_151206.pth
2026-01-31 15:12:06,324 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.264330
2026-01-31 15:12:08,770 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10254_delta_0.2617_20260131_151208.pth
2026-01-31 15:12:08,770 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.261674
2026-01-31 15:12:10,700 - Crystallization - INFO - Época 10275 | Loss: 1161638707200.000000 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2596 → 0.2617 | α: 1.35 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:11,250 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10281_delta_0.2590_20260131_151211.pth
2026-01-31 15:12:11,250 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.259037
2026-01-31 15:12:13,704 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10308_delta_0.2564_20260131_151213.pth
2026-01-31 15:12:13,704 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.256417
2026-01-31 15:12:15,290 - Crystallization - INFO - Época 10325 | Loss: 1040840969420.800049 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2548 → 0.2564 | α: 1.37 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:16,267 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10335_delta_0.2538_20260131_151216.pth
2026-01-31 15:12:16,268 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.253815
2026-01-31 15:12:18,871 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10362_delta_0.2512_20260131_151218.pth
2026-01-31 15:12:18,872 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.251230
2026-01-31 15:12:20,072 - Crystallization - INFO - Época 10375 | Loss: 941711740108.800049 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2500 → 0.2512 | α: 1.39 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:21,360 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10389_delta_0.2487_20260131_151221.pth
2026-01-31 15:12:21,360 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.248663
2026-01-31 15:12:23,821 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10416_delta_0.2461_20260131_151223.pth
2026-01-31 15:12:23,822 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.246113
2026-01-31 15:12:24,867 - Crystallization - INFO - Época 10425 | Loss: 863087283404.800049 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2453 → 0.2461 | α: 1.41 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:26,579 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10443_delta_0.2436_20260131_151226.pth
2026-01-31 15:12:26,579 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.243580
2026-01-31 15:12:29,089 - Crystallization - INFO - Checkpoint cristalino guardado: crystal_checkpoints/crystal_epoch_10470_delta_0.2411_20260131_151229.pth
2026-01-31 15:12:29,089 - Crystallization - INFO - CRISTALIZACIÓN NO COMPLETADA δ = 0.241064
2026-01-31 15:12:29,552 - Crystallization - INFO - Época 10475 | Loss: 799116872908.800049 | MSE: 0.003906 | Quant: 0.0000 | ValAcc: 1.0000 | δ: 0.2406 → 0.2411 | α: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | λ: 75047317648499552.000
2026-01-31 15:12:29,552 - Crystallization - INFO - ============================================================
2026-01-31 15:12:29,552 - Crystallization - INFO - Refinamiento completado. Mejor δ alcanzado: 0.241064
2026-01-31 15:12:29,553 - Crystallization - INFO - Mejora total: 1.90x
2026-01-31 15:12:29,553 - Crystallization - INFO - ============================================================

Resultados guardados en: results/crystallization_20260131_151229.json
Éxito: False
Mejora en δ: 1.90x
Épocas: 5500


❯ python3 precision.py
2026-01-31 15:35:08,977 - Continuation - INFO - Continuing from: crystal_checkpoints/crystal_epoch_10470_delta_0.2411_20260131_151229.pth
2026-01-31 15:35:08,987 - Continuation - INFO - Loaded checkpoint - Epoch: 10470, ValAcc: N/A, Delta: 0.2410644143819809
2026-01-31 15:35:09,578 - Continuation - INFO - Initial lambda: 7.504732e+16
2026-01-31 15:35:09,608 - Continuation - INFO - Starting epoch: 10470, delta: 0.241064
2026-01-31 15:35:13,354 - Continuation - INFO - Epoch 10520 | Loss: 7.435936e+11 | MSE: 0.003906 | Quant: 9.9083e-06 | ValAcc: 1.0000 | delta: 0.2410 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+16
2026-01-31 15:35:17,101 - Continuation - INFO - Epoch 10570 | Loss: 7.424481e+11 | MSE: 0.003906 | Quant: 9.8931e-06 | ValAcc: 1.0000 | delta: 0.2409 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+16
2026-01-31 15:35:17,174 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:35:17,175 - Continuation - INFO - New lambda: 7.504732e+16 -> 7.504732e+17
2026-01-31 15:35:20,958 - Continuation - INFO - Epoch 10620 | Loss: 6.598241e+12 | MSE: 0.003906 | Quant: 8.7921e-06 | ValAcc: 1.0000 | delta: 0.2408 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+17
2026-01-31 15:35:25,883 - Continuation - INFO - Epoch 10670 | Loss: 6.588336e+12 | MSE: 0.003906 | Quant: 8.7789e-06 | ValAcc: 1.0000 | delta: 0.2407 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+17
2026-01-31 15:35:26,088 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:35:26,088 - Continuation - INFO - New lambda: 7.504732e+17 -> 7.504732e+18
2026-01-31 15:35:31,009 - Continuation - INFO - Epoch 10720 | Loss: 6.577759e+13 | MSE: 0.003906 | Quant: 8.7648e-06 | ValAcc: 1.0000 | delta: 0.2406 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+18
2026-01-31 15:35:35,935 - Continuation - INFO - Epoch 10770 | Loss: 6.568835e+13 | MSE: 0.003906 | Quant: 8.7529e-06 | ValAcc: 1.0000 | delta: 0.2406 -> 0.2411 | alpha: 1.42 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+18
2026-01-31 15:35:36,228 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:35:36,229 - Continuation - INFO - New lambda: 7.504732e+18 -> 7.504732e+19
2026-01-31 15:35:40,811 - Continuation - INFO - Epoch 10820 | Loss: 6.559500e+14 | MSE: 0.003906 | Quant: 8.7405e-06 | ValAcc: 1.0000 | delta: 0.2405 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+19
2026-01-31 15:35:45,722 - Continuation - INFO - Epoch 10870 | Loss: 6.551445e+14 | MSE: 0.003906 | Quant: 8.7298e-06 | ValAcc: 1.0000 | delta: 0.2404 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+19
2026-01-31 15:35:46,118 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:35:46,119 - Continuation - INFO - New lambda: 7.504732e+19 -> 7.504732e+20
2026-01-31 15:35:50,646 - Continuation - INFO - Epoch 10920 | Loss: 6.543214e+15 | MSE: 0.003906 | Quant: 8.7188e-06 | ValAcc: 1.0000 | delta: 0.2404 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+20
2026-01-31 15:35:55,738 - Continuation - INFO - Epoch 500: Pruned 57 parameters
2026-01-31 15:35:55,841 - Continuation - INFO - Epoch 10970 | Loss: 6.535796e+15 | MSE: 0.003906 | Quant: 8.7089e-06 | ValAcc: 1.0000 | delta: 0.2403 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+20
2026-01-31 15:35:55,846 - Continuation - INFO - Crystal checkpoint saved: crystal_checkpoints/crystal_epoch_10970_delta_0.2403_20260131_153555_841521.pth
2026-01-31 15:35:56,350 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:35:56,350 - Continuation - INFO - New lambda: 7.504732e+20 -> 7.504732e+21
2026-01-31 15:36:00,802 - Continuation - INFO - Epoch 11020 | Loss: 6.528521e+16 | MSE: 0.003906 | Quant: 8.6992e-06 | ValAcc: 1.0000 | delta: 0.2403 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+21
2026-01-31 15:36:05,825 - Continuation - INFO - Epoch 11070 | Loss: 6.521871e+16 | MSE: 0.003906 | Quant: 8.6903e-06 | ValAcc: 1.0000 | delta: 0.2402 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+21
2026-01-31 15:36:06,427 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:36:06,428 - Continuation - INFO - New lambda: 7.504732e+21 -> 7.504732e+22
2026-01-31 15:36:10,736 - Continuation - INFO - Epoch 11120 | Loss: 6.557339e+17 | MSE: 0.003906 | Quant: 8.7376e-06 | ValAcc: 1.0000 | delta: 0.2402 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+22
2026-01-31 15:36:16,032 - Continuation - INFO - Epoch 11170 | Loss: 6.556468e+17 | MSE: 0.003906 | Quant: 8.7364e-06 | ValAcc: 1.0000 | delta: 0.2401 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+22
2026-01-31 15:36:16,721 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:36:16,721 - Continuation - INFO - New lambda: 7.504732e+22 -> 7.504732e+23
2026-01-31 15:36:22,004 - Continuation - INFO - Epoch 11220 | Loss: 6.555668e+18 | MSE: 0.003906 | Quant: 8.7354e-06 | ValAcc: 1.0000 | delta: 0.2401 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+23
2026-01-31 15:36:28,230 - Continuation - INFO - Epoch 11270 | Loss: 6.554882e+18 | MSE: 0.003906 | Quant: 8.7343e-06 | ValAcc: 1.0000 | delta: 0.2401 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+23
2026-01-31 15:36:29,246 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:36:29,247 - Continuation - INFO - New lambda: 7.504732e+23 -> 7.504732e+24
2026-01-31 15:36:35,500 - Continuation - INFO - Epoch 11320 | Loss: 6.554164e+19 | MSE: 0.003906 | Quant: 8.7334e-06 | ValAcc: 1.0000 | delta: 0.2400 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+24
2026-01-31 15:36:43,356 - Continuation - INFO - Epoch 11370 | Loss: 6.553459e+19 | MSE: 0.003906 | Quant: 8.7324e-06 | ValAcc: 1.0000 | delta: 0.2400 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+24
2026-01-31 15:36:44,688 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:36:44,689 - Continuation - INFO - New lambda: 7.504732e+24 -> 7.504732e+25
2026-01-31 15:36:53,571 - Continuation - INFO - Epoch 11420 | Loss: 6.552811e+20 | MSE: 0.003906 | Quant: 8.7316e-06 | ValAcc: 1.0000 | delta: 0.2400 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+25
2026-01-31 15:37:06,120 - Continuation - INFO - Epoch 1000: Pruned 62 parameters
2026-01-31 15:37:06,287 - Continuation - INFO - Epoch 11470 | Loss: 5.413088e+20 | MSE: 0.003906 | Quant: 7.2129e-06 | ValAcc: 1.0000 | delta: 0.2400 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+25
2026-01-31 15:37:06,292 - Continuation - INFO - Crystal checkpoint saved: crystal_checkpoints/crystal_epoch_11470_delta_0.2400_20260131_153706_287283.pth
2026-01-31 15:37:08,496 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:37:08,496 - Continuation - INFO - New lambda: 7.504732e+25 -> 7.504732e+26
2026-01-31 15:37:15,499 - Continuation - INFO - Epoch 11520 | Loss: 5.412606e+21 | MSE: 0.003906 | Quant: 7.2123e-06 | ValAcc: 1.0000 | delta: 0.2400 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+26
2026-01-31 15:37:24,166 - Continuation - INFO - Epoch 11570 | Loss: 5.412136e+21 | MSE: 0.003906 | Quant: 7.2116e-06 | ValAcc: 1.0000 | delta: 0.2399 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+26
2026-01-31 15:37:26,058 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:37:26,059 - Continuation - INFO - New lambda: 7.504732e+26 -> 7.504732e+27
2026-01-31 15:37:32,843 - Continuation - INFO - Epoch 11620 | Loss: 5.411696e+22 | MSE: 0.003906 | Quant: 7.2110e-06 | ValAcc: 1.0000 | delta: 0.2399 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+27
2026-01-31 15:37:41,459 - Continuation - INFO - Epoch 11670 | Loss: 5.411265e+22 | MSE: 0.003906 | Quant: 7.2105e-06 | ValAcc: 1.0000 | delta: 0.2399 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+27
2026-01-31 15:37:43,500 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:37:43,500 - Continuation - INFO - New lambda: 7.504732e+27 -> 7.504732e+28
2026-01-31 15:37:48,278 - Continuation - INFO - Epoch 11720 | Loss: 5.410871e+23 | MSE: 0.003906 | Quant: 7.2099e-06 | ValAcc: 1.0000 | delta: 0.2399 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+28
2026-01-31 15:37:54,648 - Continuation - INFO - Epoch 11770 | Loss: 5.410490e+23 | MSE: 0.003906 | Quant: 7.2094e-06 | ValAcc: 1.0000 | delta: 0.2399 -> 0.2411 | alpha: 1.43 | Sparsity: 0.00% | Cryst: 100.00% | lambda: 7.504732e+28
2026-01-31 15:37:56,293 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:37:56,293 - Continuation - INFO - New lambda: 7.504732e+28 -> 7.504732e+29
2026-01-31 15:37:59,918 - Continuation - INFO - Epoch 11820 | Loss: 5.410141e+24 | MSE: 0.003906 | Quant: 7.2090e-06 | ValAcc: 1.0000 | delta: 0.2398 -> 0.2411 | alpha: 1.43 | Sparsity: 0.01% | Cryst: 100.00% | lambda: 7.504732e+29
2026-01-31 15:38:05,215 - Continuation - INFO - Epoch 11870 | Loss: 5.409803e+24 | MSE: 0.003906 | Quant: 7.2085e-06 | ValAcc: 1.0000 | delta: 0.2398 -> 0.2411 | alpha: 1.43 | Sparsity: 0.01% | Cryst: 100.00% | lambda: 7.504732e+29
2026-01-31 15:38:06,752 - Continuation - INFO - No improvement for 101 epochs. Growing lambda...
2026-01-31 15:38:06,752 - Continuation - INFO - New lambda: 7.504732e+29 -> 7.504732e+30
2026-01-31 15:38:06,999 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11886
2026-01-31 15:38:07,244 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11888
2026-01-31 15:38:07,518 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11890
2026-01-31 15:38:07,754 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11892
2026-01-31 15:38:07,990 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11894
2026-01-31 15:38:08,206 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11896
2026-01-31 15:38:08,414 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11898
2026-01-31 15:38:08,622 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11900
2026-01-31 15:38:08,884 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11902
2026-01-31 15:38:09,167 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11904
2026-01-31 15:38:09,439 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11906
2026-01-31 15:38:09,694 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11908
2026-01-31 15:38:09,936 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11910
2026-01-31 15:38:10,149 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11912
2026-01-31 15:38:10,361 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11914
2026-01-31 15:38:10,562 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11916
2026-01-31 15:38:10,770 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11918
2026-01-31 15:38:10,972 - Continuation - INFO - Epoch 11920 | Loss: 5.409489e+25 | MSE: 0.003906 | Quant: 7.2081e-06 | ValAcc: 1.0000 | delta: 0.2398 -> 0.2411 | alpha: 1.43 | Sparsity: 0.01% | Cryst: 100.00% | lambda: 7.504732e+30
2026-01-31 15:38:10,978 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11920
2026-01-31 15:38:11,189 - Continuation - INFO - [DANGER ZONE] Latest checkpoint updated: epoch 11922


```
