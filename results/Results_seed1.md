# Results control seed 1

```text
 python3 experiment2.py --seed 1
2026-01-30 20:02:27,954 - SingleExperimentRunner - INFO - Starting single experiment with seed 1
2026-01-30 20:02:28,597 - SingleExperimentRunner - INFO - Starting training for 5000 epochs
/home/grisun0/src/py/HPU-Core/experiment2.py:285: UserWarning: cov(): degrees of freedom is <= 0. Correction should be strictly less than the number of observations. (Triggered internally at /pytorch/aten/src/ATen/native/Correlation.cpp:116.)
  correlation_matrix = torch.corrcoef(weights)
2026-01-30 20:04:40,031 - SingleExperimentRunner - INFO - Epoch   10: Loss=0.005216, ValLoss=0.004878, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:06:51,114 - SingleExperimentRunner - INFO - Epoch   20: Loss=0.003915, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:07:29,881 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_23_20260130_200729.pth
2026-01-30 20:08:59,804 - SingleExperimentRunner - INFO - Epoch   30: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4991, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:11:02,751 - SingleExperimentRunner - INFO - Epoch   40: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4993, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:12:41,195 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_48_20260130_201241.pth
2026-01-30 20:13:05,730 - SingleExperimentRunner - INFO - Epoch   50: Loss=0.003906, ValLoss=0.003906, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4995, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:15:08,450 - SingleExperimentRunner - INFO - Epoch   60: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4998, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:17:11,757 - SingleExperimentRunner - INFO - Epoch   70: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.5000, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:17:48,658 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_73_20260130_201748.pth
2026-01-30 20:18:59,690 - SingleExperimentRunner - INFO - Epoch   80: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4997, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:20:36,292 - SingleExperimentRunner - INFO - Epoch   90: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4997, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:22:12,764 - SingleExperimentRunner - INFO - Epoch  100: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4996, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:22:51,381 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_104_20260130_202251.pth
2026-01-30 20:23:49,298 - SingleExperimentRunner - INFO - Epoch  110: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4996, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:25:26,972 - SingleExperimentRunner - INFO - Epoch  120: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4996, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:27:30,606 - SingleExperimentRunner - INFO - Epoch  130: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4995, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:27:55,084 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_132_20260130_202755.pth
2026-01-30 20:29:35,846 - SingleExperimentRunner - INFO - Epoch  140: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4994, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:32:01,092 - SingleExperimentRunner - INFO - Epoch  150: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4994, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:32:59,196 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_154_20260130_203259.pth
2026-01-30 20:34:26,370 - SingleExperimentRunner - INFO - Epoch  160: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4994, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:36:49,266 - SingleExperimentRunner - INFO - Epoch  170: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4993, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:38:03,331 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_176_20260130_203803.pth
2026-01-30 20:38:52,707 - SingleExperimentRunner - INFO - Epoch  180: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4993, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:40:57,070 - SingleExperimentRunner - INFO - Epoch  190: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4993, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:43:00,306 - SingleExperimentRunner - INFO - Epoch  200: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4993, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:43:12,576 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_201_20260130_204312.pth
2026-01-30 20:45:03,478 - SingleExperimentRunner - INFO - Epoch  210: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4992, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:47:02,210 - SingleExperimentRunner - INFO - Epoch  220: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4992, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:48:19,820 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_228_20260130_204819.pth
2026-01-30 20:48:39,216 - SingleExperimentRunner - INFO - Epoch  230: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.69, Kappa=inf, Delta=0.4991, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:50:16,184 - SingleExperimentRunner - INFO - Epoch  240: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4990, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:51:53,032 - SingleExperimentRunner - INFO - Epoch  250: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4990, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:53:20,109 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_259_20260130_205320.pth
2026-01-30 20:53:29,779 - SingleExperimentRunner - INFO - Epoch  260: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4989, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:55:06,377 - SingleExperimentRunner - INFO - Epoch  270: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4989, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:56:43,173 - SingleExperimentRunner - INFO - Epoch  280: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4989, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:58:20,845 - SingleExperimentRunner - INFO - Epoch  290: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 20:58:20,854 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_290_20260130_205820.pth
2026-01-30 21:00:22,215 - SingleExperimentRunner - INFO - Epoch  300: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:02:32,497 - SingleExperimentRunner - INFO - Epoch  310: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:03:28,488 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_314_20260130_210328.pth
2026-01-30 21:04:51,312 - SingleExperimentRunner - INFO - Epoch  320: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:07:13,052 - SingleExperimentRunner - INFO - Epoch  330: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:08:31,738 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_336_20260130_210831.pth
2026-01-30 21:09:25,026 - SingleExperimentRunner - INFO - Epoch  340: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4988, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:11:29,975 - SingleExperimentRunner - INFO - Epoch  350: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:13:34,151 - SingleExperimentRunner - INFO - Epoch  360: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:13:34,160 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_360_20260130_211334.pth
2026-01-30 21:15:41,311 - SingleExperimentRunner - INFO - Epoch  370: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:17:50,615 - SingleExperimentRunner - INFO - Epoch  380: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:18:43,779 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_384_20260130_211843.pth
2026-01-30 21:19:59,056 - SingleExperimentRunner - INFO - Epoch  390: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:22:07,810 - SingleExperimentRunner - INFO - Epoch  400: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4987, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:23:50,414 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_408_20260130_212350.pth
2026-01-30 21:24:14,963 - SingleExperimentRunner - INFO - Epoch  410: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:26:33,480 - SingleExperimentRunner - INFO - Epoch  420: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:28:53,650 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_425_20260130_212853.pth
2026-01-30 21:31:12,985 - SingleExperimentRunner - INFO - Epoch  430: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:33:57,139 - SingleExperimentRunner - INFO - Checkpoint saved: checkpoints/checkpoint_epoch_437_20260130_213357.pth
2026-01-30 21:35:02,949 - SingleExperimentRunner - INFO - Epoch  440: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00
2026-01-30 21:38:53,062 - SingleExperimentRunner - INFO - Epoch  450: Loss=0.003906, ValLoss=0.003905, ValAcc=1.0000, LC=1.0000, SP=0.0000, Alpha=0.70, Kappa=inf, Delta=0.4986, Temp=0.00e+00, Cv=0.00e+00

```
