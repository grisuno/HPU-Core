#!/usr/bin/env python3
"""
Script de Refinamiento Cristalino para Discretización de Pesos
Carga un checkpoint estable y fuerza la transición vidrio -> cristal puro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, Any
import logging
from collections import deque

# Importar tus clases existentes
try:
    from experiment2 import (
        Config, HamiltonianNeuralNetwork, HamiltonianDataset, 
        CrystallographyMetricsCalculator, LoggerFactory, SeedManager,
        LocalComplexityAnalyzer, SuperpositionAnalyzer
    )
except ImportError as e:
    print(f"Error importando desde experiment2: {e}")
    raise


class CrystallizationConfig:
    """Configuración agresiva para forzar discretización"""
    # Checkpoint a cargar
    CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_1009_20260128_133022.pth"
    
    # Hiperparámetros agresivos
    LEARNING_RATE = 1e-5           # Ligeramente mayor para evitar estancamiento
    WEIGHT_DECAY = 1e-2            # Alto para empujar a cero
    QUANTIZATION_LAMBDA = 0.5      # Fuerza inicial de la pérdida de cuantización
    QUANTIZATION_RAMP = True       # Incrementar lambda progresivamente
    
    PRUNING_THRESHOLD = 0.05       # Umbral inicial para poda
    PRUNING_SCHEDULE = [0.1, 0.2, 0.3, 0.4]  # Umbrales progresivos
    
    # Criterios de parada
    TARGET_DELTA = 0.05            # Objetivo: δ < 0.05 (muy cristalino)
    MAX_EPOCHS = 5500
    PATIENCE = 100                 # Épocas sin mejora antes de aumentar agresividad
    
    # Validación
    MAINTAIN_ACCURACY = True       # Si True, no permitir bajar de ValAcc=1.0
    ACCURACY_TOLERANCE = 0.01      # Tolerancia para caída de accuracy


class CrystallizationLoss(nn.Module):
    """
    Pérdida combinada: MSE + penalización de cuantización
    Fuerza los pesos a caer en {-1, 0, 1}
    """
    def __init__(self, lambda_quant: float = CrystallizationConfig.QUANTIZATION_LAMBDA):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_quant = lambda_quant
        
    def quantization_penalty(self, model: nn.Module) -> torch.Tensor:
        """Penalización L2 de la distancia al entero más cercano"""
        penalty = 0.0
        total_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                # Distancia al redondeo más cercano
                rounded = torch.round(param)
                penalty += torch.sum(torch.abs(param - rounded))
                total_params += param.numel()
        
        return penalty / (total_params + 1e-8)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                model: nn.Module) -> tuple[torch.Tensor, dict]:
        mse_loss = self.mse(predictions, targets)
        quant_loss = self.quantization_penalty(model)
        total_loss = mse_loss + self.lambda_quant * quant_loss
        
        metrics = {
            'mse': mse_loss.item(),
            'quant_penalty': quant_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, metrics


class StructuralPruner:
    """Implementa poda progresiva de pesos pequeños"""
    def __init__(self, thresholds: list = CrystallizationConfig.PRUNING_SCHEDULE):
        self.thresholds = thresholds
        self.current_stage = 0
        self.pruned_count = 0
        
    def should_prune(self, epoch: int) -> bool:
        """Determina si es momento de podar (cada 500 épocas)"""
        return epoch > 0 and epoch % 500 == 0 and self.current_stage < len(self.thresholds)
    
    def prune(self, model: nn.Module, force_threshold: float = None) -> int:
        """
        Poda pesos con |w| < threshold
        Retorna número de parámetros podados
        """
        if self.current_stage >= len(self.thresholds) and force_threshold is None:
            return 0
            
        threshold = force_threshold if force_threshold is not None else self.thresholds[self.current_stage]
        pruned = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    mask = torch.abs(param) < threshold
                    pruned += mask.sum().item()
                    param.data[mask] = 0.0
        
        if force_threshold is None:
            self.current_stage += 1
            
        self.pruned_count += pruned
        return pruned
    
    def get_sparsity(self, model: nn.Module) -> float:
        """Calcula porcentaje de pesos exactamente en cero"""
        total = 0
        zeros = 0
        
        with torch.no_grad():
            for param in model.parameters():
                total += param.numel()
                zeros += (param == 0).sum().item()
                
        return zeros / total if total > 0 else 0.0


class CrystallizationEngine:
    """
    Motor de refinamiento que carga un checkpoint y fuerza discretización
    """
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = self._setup_logger()
        
        # Cargar checkpoint
        self.model, self.start_epoch, self.initial_metrics = self._load_checkpoint()
        
        # Crear NUEVO optimizador (no cargar estado anterior para evitar conflictos)
        # Usamos SGD con momentum como en tu entrenamiento original, pero con LR bajo
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=CrystallizationConfig.LEARNING_RATE,
            momentum=0.9,
            weight_decay=CrystallizationConfig.WEIGHT_DECAY
        )
        
        self.logger.info(f"Optimizador creado: SGD lr={CrystallizationConfig.LEARNING_RATE}, "
                        f"wd={CrystallizationConfig.WEIGHT_DECAY}")
        
        self.criterion = CrystallizationLoss()
        self.pruner = StructuralPruner()
        self.metrics_history = []
        
        # Dataset para validación
        self.dataset = HamiltonianDataset(
            num_samples=Config.NUM_SAMPLES,
            grid_size=Config.GRID_SIZE
        )
        self.val_x, self.val_y = self.dataset.get_validation_batch()
        self.val_x = self.val_x.to(self.device)
        self.val_y = self.val_y.to(self.device)
        
        # Métricas iniciales
        self.best_delta = self.initial_metrics.get('delta', 1.0)
        self.best_state = None
        self.lambda_history = [CrystallizationConfig.QUANTIZATION_LAMBDA]
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("Crystallization")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_checkpoint(self) -> tuple:
        """Carga el checkpoint y retorna modelo, época y métricas"""
        self.logger.info(f"Cargando checkpoint: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Crear modelo
        model = HamiltonianNeuralNetwork(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(self.device)
        
        # Cargar pesos
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Extraer información
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        # Si no hay métricas, calcular delta manualmente
        if not metrics or 'delta' not in metrics:
            metrics = self._compute_initial_metrics(model)
        
        self.logger.info(f"Checkpoint cargado - Época: {epoch}, "
                        f"ValAcc: {metrics.get('val_acc', 'N/A'):.4f}, "
                        f"Delta: {metrics.get('delta', 'N/A'):.4f}")
        
        return model, epoch, metrics
    
    def _compute_initial_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Calcula métricas iniciales si no vienen en el checkpoint"""
        model.eval()
        with torch.no_grad():
            outputs = model(self.val_x)
            mse_per_sample = ((outputs - self.val_y) ** 2).mean(dim=(1, 2))
            val_acc = (mse_per_sample < Config.MSE_THRESHOLD).float().mean().item()
        
        # Calcular delta manualmente
        coeffs = {name: param.data.clone() for name, param in model.named_parameters()}
        deltas = []
        for tensor in coeffs.values():
            if tensor.numel() > 0:
                delta = (tensor - tensor.round()).abs().max().item()
                deltas.append(delta)
        delta = max(deltas) if deltas else 1.0
        
        return {'val_acc': val_acc, 'delta': delta}
    
    def compute_discretization_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de cristalinidad actuales"""
        coeffs = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Delta (margen de discretización)
        deltas = []
        for tensor in coeffs.values():
            if tensor.numel() > 0:
                delta = (tensor - tensor.round()).abs().max().item()
                deltas.append(delta)
        delta = max(deltas) if deltas else 1.0
        
        # Alpha (pureza)
        alpha = -np.log(delta + 1e-15) if delta > 1e-10 else 20.0
        
        # Histograma de pesos
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        # Contar pesos cerca de enteros
        near_zero = (all_weights.abs() < 0.1).sum().item()
        near_one = ((all_weights - 1).abs() < 0.1).sum().item()
        near_minus_one = ((all_weights + 1).abs() < 0.1).sum().item()
        total = all_weights.numel()
        
        return {
            'delta': delta,
            'alpha': alpha,
            'purity_index': 1.0 - delta,
            'is_crystal': alpha > 7.0,
            'sparsity': self.pruner.get_sparsity(self.model),
            'weight_distribution': {
                'near_zero': near_zero / total,
                'near_one': near_one / total,
                'near_minus_one': near_minus_one / total,
                'crystallized': (near_zero + near_one + near_minus_one) / total
            }
        }
    
    def validate(self) -> tuple[float, float]:
        """Valida el modelo manteniendo accuracy"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.val_x)
            mse = F.mse_loss(outputs, self.val_y).item()
            # Accuracy basada en umbral
            mse_per_sample = ((outputs - self.val_y) ** 2).mean(dim=(1, 2))
            acc = (mse_per_sample < Config.MSE_THRESHOLD).float().mean().item()
        return mse, acc
    
    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Entrena una época con pérdida de cuantización"""
        self.model.train()
        
        # Poda programada
        if self.pruner.should_prune(epoch):
            pruned = self.pruner.prune(self.model)
            self.logger.info(f"Época {epoch}: Podados {pruned:,} parámetros")
        
        # Entrenamiento simple (un batch del dataset)
        train_loader = DataLoader(self.dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        total_loss = 0
        mse_total = 0
        quant_total = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss, metrics = self.criterion(outputs, batch_y, self.model)
            loss.backward()
            
            # Clip de gradientes suave
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += metrics['total']
            mse_total += metrics['mse']
            quant_total += metrics['quant_penalty']
            n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches if n_batches > 0 else 0,
            'mse': mse_total / n_batches if n_batches > 0 else 0,
            'quant_penalty': quant_total / n_batches if n_batches > 0 else 0
        }
    
    def refine(self) -> Dict[str, Any]:
        """
        Ejecuta el refinamiento hasta alcanzar δ < TARGET_DELTA o MAX_EPOCHS
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO CRISTALIZACIÓN FORZADA")
        self.logger.info(f"Objetivo: δ < {CrystallizationConfig.TARGET_DELTA}")
        self.logger.info(f"LR: {CrystallizationConfig.LEARNING_RATE}, "
                        f"WD: {CrystallizationConfig.WEIGHT_DECAY}")
        self.logger.info(f"Lambda quant inicial: {self.criterion.lambda_quant}")
        self.logger.info("=" * 60)
        
        patience_counter = 0
        last_improvement_epoch = 0
        
        for epoch in range(1, CrystallizationConfig.MAX_EPOCHS + 1):
            # Entrenar
            train_metrics = self.train_epoch(epoch)
            val_mse, val_acc = self.validate()
            cryst_metrics = self.compute_discretization_metrics()
            
            current_delta = cryst_metrics['delta']
            
            # Guardar historial
            record = {
                'epoch': self.start_epoch + epoch,
                'train_loss': train_metrics['total_loss'],
                'mse': train_metrics['mse'],
                'quant_penalty': train_metrics['quant_penalty'],
                'val_mse': val_mse,
                'val_acc': val_acc,
                'delta': current_delta,
                'alpha': cryst_metrics['alpha'],
                'sparsity': cryst_metrics['sparsity'],
                'crystallized_ratio': cryst_metrics['weight_distribution']['crystallized'],
                'lambda_quant': self.criterion.lambda_quant
            }
            self.metrics_history.append(record)
            
            # Logging cada 50 épocas
            if epoch % 50 == 0:
                self.logger.info(
                    f"Época {self.start_epoch + epoch:4d} | "
                    f"Loss: {train_metrics['total_loss']:.6f} | "
                    f"MSE: {train_metrics['mse']:.6f} | "
                    f"Quant: {train_metrics['quant_penalty']:.4f} | "
                    f"ValAcc: {val_acc:.4f} | "
                    f"δ: {current_delta:.4f} → {self.best_delta:.4f} | "
                    f"α: {cryst_metrics['alpha']:.2f} | "
                    f"Sparsity: {cryst_metrics['sparsity']:.2%} | "
                    f"Cryst: {cryst_metrics['weight_distribution']['crystallized']:.2%} | "
                    f"λ: {self.criterion.lambda_quant:.3f}"
                )
            
            # Verificar si mejoramos el delta
            improved = current_delta < self.best_delta * 0.99  # Mejora del 1%
            if improved:
                improvement = (self.best_delta - current_delta) / self.best_delta * 100
                self.best_delta = current_delta
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                last_improvement_epoch = epoch
                
                # Guardar checkpoint si es el mejor hasta ahora
                if epoch % 100 == 0 or current_delta < 0.1:
                    self._save_crystal_checkpoint(epoch, cryst_metrics, val_acc)
            else:
                patience_counter += 1
            
            # Verificar objetivo alcanzado
            if current_delta < CrystallizationConfig.TARGET_DELTA:
                self.logger.info("=" * 60)
                self.logger.info(f"¡CRISTALIZACIÓN COMPLETADA! δ = {current_delta:.6f}")
                self.logger.info("=" * 60)
                self._save_crystal_checkpoint(epoch, cryst_metrics, val_acc, final=True)
                return self._compile_results(success=True, final_epoch=epoch)
            
            # Early stopping por paciencia - aumentar agresividad
            if patience_counter > CrystallizationConfig.PATIENCE:
                epochs_without = epoch - last_improvement_epoch
                self.logger.info(f"Sin mejora en {epochs_without} épocas. Aumentando agresividad...")
                
                # Aumentar lambda de cuantización
                old_lambda = self.criterion.lambda_quant
                self.criterion.lambda_quant *= 1.5
                self.lambda_history.append(self.criterion.lambda_quant)
                
                # Reducir LR ligeramente
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 5e-4  # Sube a un valor que genere movimiento
                self.criterion.lambda_quant *= 2.0
                
                # Forzar poda agresiva
                self.pruner.prune(self.model, force_threshold=0.15)
                
                self.logger.info(f"Nueva λ: {old_lambda:.3f} → {self.criterion.lambda_quant:.3f}, "
                               f"LR reducido en 10%")
                patience_counter = 0
            
            # Verificar que no perdimos accuracy si es requisito
            if (CrystallizationConfig.MAINTAIN_ACCURACY and 
                val_acc < 1.0 - CrystallizationConfig.ACCURACY_TOLERANCE):
                self.logger.warning(f"Accuracy cayó a {val_acc:.4f}. Restaurando mejor estado...")
                if self.best_state:
                    self.model.load_state_dict(self.best_state)
                    # Reducir lambda para no ser tan agresivo
                    self.criterion.lambda_quant *= 0.8
        
        # Si terminamos sin alcanzar el objetivo
        self.logger.info("=" * 60)
        self.logger.info(f"Refinamiento completado. Mejor δ alcanzado: {self.best_delta:.6f}")
        self.logger.info(f"Mejora total: {(self.initial_metrics.get('delta', 1.0) / self.best_delta):.2f}x")
        self.logger.info("=" * 60)
        return self._compile_results(success=False, final_epoch=CrystallizationConfig.MAX_EPOCHS)
    
    def _save_crystal_checkpoint(self, epoch: int, metrics: Dict, val_acc: float, final: bool = False):
        """Guarda checkpoint cristalino"""
        os.makedirs("crystal_checkpoints", exist_ok=True)
        
        suffix = "FINAL" if final else f"epoch_{self.start_epoch + epoch}_delta_{metrics['delta']:.4f}"
        path = f"crystal_checkpoints/crystal_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        torch.save({
            'epoch': self.start_epoch + epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'val_acc': val_acc,
            'config': {
                'lr': CrystallizationConfig.LEARNING_RATE,
                'weight_decay': CrystallizationConfig.WEIGHT_DECAY,
                'quant_lambda': self.criterion.lambda_quant,
                'lambda_history': self.lambda_history
            }
        }, path)
        
        self.logger.info(f"Checkpoint cristalino guardado: {path}")
    
    def _compile_results(self, success: bool, final_epoch: int) -> Dict[str, Any]:
        """Compila resultados finales"""
        initial_delta = self.initial_metrics.get('delta', 1.0)
        return {
            'success': success,
            'initial_delta': initial_delta,
            'final_delta': self.best_delta,
            'improvement_ratio': initial_delta / (self.best_delta + 1e-10),
            'total_epochs': final_epoch,
            'final_sparsity': self.pruner.get_sparsity(self.model),
            'lambda_history': self.lambda_history,
            'metrics_history': self.metrics_history,
            'timestamp': datetime.now().isoformat()
        }


def analyze_discretization(checkpoint_path: str):
    """
    Análisis detallado de la discretización de un checkpoint
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE DISCRETIZACIÓN")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Época: {checkpoint.get('epoch', 'N/A')}")
    
    all_weights = []
    layer_stats = {}
    
    for name, param in state_dict.items():
        if 'weight' not in name or param.numel() == 0:
            continue
            
        w = param.flatten().float()
        all_weights.append(w)
        
        # Estadísticas por capa
        rounded = torch.round(w)
        delta = (w - rounded).abs()
        
        layer_stats[name] = {
            'shape': list(param.shape),
            'mean': w.mean().item(),
            'std': w.std().item(),
            'delta_max': delta.max().item(),
            'delta_mean': delta.mean().item(),
            'pct_near_zero': (w.abs() < 0.1).sum().item() / w.numel() * 100,
            'pct_near_one': ((w - 1).abs() < 0.1).sum().item() / w.numel() * 100,
            'pct_near_minus_one': ((w + 1).abs() < 0.1).sum().item() / w.numel() * 100,
        }
    
    # Estadísticas globales
    all_w = torch.cat(all_weights)
    rounded = torch.round(all_w)
    global_delta = (all_w - rounded).abs()
    
    print(f"\n--- ESTADÍSTICAS GLOBALES ---")
    print(f"Total parámetros: {all_w.numel():,}")
    print(f"δ (max): {global_delta.max().item():.6f}")
    print(f"δ (mean): {global_delta.mean().item():.6f}")
    print(f"α (pureza): {-np.log(global_delta.max().item() + 1e-15):.2f}")
    
    print(f"\nDistribución de pesos:")
    print(f"  Cerca de 0: {(all_w.abs() < 0.1).sum().item() / all_w.numel() * 100:.1f}%")
    print(f"  Cerca de 1: {((all_w - 1).abs() < 0.1).sum().item() / all_w.numel() * 100:.1f}%")
    print(f"  Cerca de -1: {((all_w + 1).abs() < 0.1).sum().item() / all_w.numel() * 100:.1f}%")
    print(f"  Cristalizados: {((all_w.abs() < 0.1) | ((all_w - 1).abs() < 0.1) | ((all_w + 1).abs() < 0.1)).sum().item() / all_w.numel() * 100:.1f}%")
    
    # Histograma
    hist, bins = torch.histogram(all_w, bins=20, range=(-2, 2))
    print(f"\nHistograma (bins={len(bins)-1}):")
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / hist.max() * 30)
        print(f"  [{bins[i]:+.2f}, {bins[i+1]:+.2f}]: {bar} ({hist[i].item():,})")
    
    print(f"\n--- TOP 5 CAPAS MÁS CRISTALINAS (menor δ) ---")
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]['delta_max'])
    for name, stats in sorted_layers[:5]:
        print(f"\n{name}:")
        print(f"  δ_max: {stats['delta_max']:.6f}, δ_mean: {stats['delta_mean']:.6f}")
        print(f"  Near 0: {stats['pct_near_zero']:.1f}%, Near ±1: {stats['pct_near_one'] + stats['pct_near_minus_one']:.1f}%")
    
    print(f"\n{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Refinamiento cristalino de redes neuronales')
    parser.add_argument('--mode', choices=['refine', 'analyze'], default='refine',
                       help='Modo: refine (entrenar) o analyze (analizar checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=CrystallizationConfig.CHECKPOINT_PATH,
                       help='Path al checkpoint a refinar/analizar')
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_discretization(args.checkpoint)
    else:
        # Modo refinamiento
        engine = CrystallizationEngine(args.checkpoint)
        results = engine.refine()
        
        # Guardar resultados
        os.makedirs("results", exist_ok=True)
        results_path = f"results/crystallization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {results_path}")
        print(f"Éxito: {results['success']}")
        print(f"Mejora en δ: {results['improvement_ratio']:.2f}x")
        print(f"Épocas: {results['total_epochs']}")


if __name__ == "__main__":
    main()
