import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any
import logging
import glob

from experiment2 import (
    Config, HamiltonianNeuralNetwork, HamiltonianDataset, 
    CrystallographyMetricsCalculator, LoggerFactory, SeedManager,
    LocalComplexityAnalyzer, SuperpositionAnalyzer
)

from refinamiento import (
    CrystallizationConfig, CrystallizationLoss, StructuralPruner, 
    analyze_discretization
)


class MassiveLambdaConfig:
    CHECKPOINT_DIR = "crystal_checkpoints"
    FALLBACK_CHECKPOINT = "crystal_checkpoints/crystal_epoch_5309_delta_0.3706_20260128_150931.pth"
    MAX_EPOCHS = 7459
    TARGET_DELTA = 0.05
    
    LAMBDA_MAX = 1e300
    LAMBDA_GROWTH = 10.0
    

class CrystallizationLossMassive(nn.Module):
    def __init__(self, lambda_quant: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_quant = lambda_quant
        
    def quantization_penalty(self, model: nn.Module) -> torch.Tensor:
        penalty = torch.tensor(0.0, dtype=torch.float64)
        total_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                rounded = torch.round(param)
                penalty += torch.sum(torch.abs(param.double() - rounded.double()))
                total_params += param.numel()
        
        return (penalty / (total_params + 1e-8)).float()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                model: nn.Module) -> tuple[torch.Tensor, dict]:
        mse_loss = self.mse(predictions, targets)
        quant_loss = self.quantization_penalty(model)
        
        # Use float64 for the multiplication to avoid overflow
        total_loss = mse_loss + torch.tensor(self.lambda_quant, dtype=torch.float64) * quant_loss.double()
        total_loss = total_loss.float()
        
        metrics = {
            'mse': mse_loss.item(),
            'quant_penalty': quant_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, metrics


class ContinuationEngine:
    def __init__(self, checkpoint_path: str = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = self._setup_logger()
        
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        self.checkpoint_path = checkpoint_path
        self.logger.info(f"Continuing from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = HamiltonianNeuralNetwork(
            grid_size=Config.GRID_SIZE,
            hidden_dim=Config.HIDDEN_DIM,
            num_spectral_layers=Config.NUM_SPECTRAL_LAYERS
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.start_epoch = checkpoint.get('epoch', 0)
        
        if 'metrics' in checkpoint:
            self.initial_metrics = checkpoint['metrics']
        elif 'delta' in checkpoint:
            self.initial_metrics = {
                'delta': checkpoint.get('delta', 0.37),
                'val_acc': checkpoint.get('val_acc', 1.0)
            }
        else:
            self.initial_metrics = self._compute_initial_metrics(self.model)
        
        self.logger.info(f"Loaded checkpoint - Epoch: {self.start_epoch}, "
                        f"ValAcc: {self.initial_metrics.get('val_acc', 'N/A')}, "
                        f"Delta: {self.initial_metrics.get('delta', 'N/A')}")
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=CrystallizationConfig.LEARNING_RATE,
            momentum=0.9,
            weight_decay=CrystallizationConfig.WEIGHT_DECAY
        )
        
        config_in_checkpoint = checkpoint.get('config', {})
        initial_lambda = config_in_checkpoint.get('quant_lambda', CrystallizationConfig.QUANTIZATION_LAMBDA)
        
        # NEVER reset lambda, keep it massive
        if initial_lambda < 1e10:
            self.logger.warning(f"Lambda too small ({initial_lambda:.2e}), growing to 1e16")
            initial_lambda = 1e16
        elif initial_lambda > MassiveLambdaConfig.LAMBDA_MAX:
            self.logger.warning(f"Lambda at max capacity: {initial_lambda:.2e}")
            initial_lambda = MassiveLambdaConfig.LAMBDA_MAX
        
        self.criterion = CrystallizationLossMassive(lambda_quant=initial_lambda)
        self.lambda_history = config_in_checkpoint.get('lambda_history', [initial_lambda])
        
        self.logger.info(f"Initial lambda: {initial_lambda:.6e}")
        
        self.pruner = StructuralPruner()
        self.metrics_history = []
        
        self.dataset = HamiltonianDataset(
            num_samples=Config.NUM_SAMPLES,
            grid_size=Config.GRID_SIZE
        )
        self.val_x, self.val_y = self.dataset.get_validation_batch()
        self.val_x = self.val_x.to(self.device)
        self.val_y = self.val_y.to(self.device)
        
        self.best_delta = self.initial_metrics.get('delta', 1.0)
        self.best_state = None
        self.epochs_without_improvement = 0
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("Continuation")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _find_latest_checkpoint(self) -> str:
        if not os.path.exists(MassiveLambdaConfig.CHECKPOINT_DIR):
            return MassiveLambdaConfig.FALLBACK_CHECKPOINT
        
        pattern = os.path.join(MassiveLambdaConfig.CHECKPOINT_DIR, "crystal_*.pth")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return MassiveLambdaConfig.FALLBACK_CHECKPOINT
        
        best_checkpoint = None
        best_delta = float('inf')
        
        for cp in checkpoints:
            try:
                if 'delta_' in cp:
                    delta_str = cp.split('delta_')[1].split('_')[0].replace('.pth', '')
                    delta = float(delta_str)
                    if delta < best_delta:
                        best_delta = delta
                        best_checkpoint = cp
            except:
                continue
        
        if best_checkpoint is None:
            best_checkpoint = max(checkpoints, key=os.path.getmtime)
        
        return best_checkpoint
    
    def _compute_initial_metrics(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            outputs = model(self.val_x)
            mse_per_sample = ((outputs - self.val_y) ** 2).mean(dim=(1, 2))
            val_acc = (mse_per_sample < Config.MSE_THRESHOLD).float().mean().item()
        
        coeffs = {name: param.data.clone() for name, param in model.named_parameters()}
        deltas = []
        for tensor in coeffs.values():
            if tensor.numel() > 0:
                delta = (tensor - tensor.round()).abs().max().item()
                deltas.append(delta)
        delta = max(deltas) if deltas else 1.0
        
        return {'val_acc': val_acc, 'delta': delta}
    
    def compute_discretization_metrics(self) -> Dict[str, Any]:
        coeffs = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        deltas = []
        for tensor in coeffs.values():
            if tensor.numel() > 0:
                delta = (tensor - tensor.round()).abs().max().item()
                deltas.append(delta)
        delta = max(deltas) if deltas else 1.0
        
        alpha = -np.log(delta + 1e-15) if delta > 1e-10 else 20.0
        
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
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
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.val_x)
            mse = F.mse_loss(outputs, self.val_y).item()
            mse_per_sample = ((outputs - self.val_y) ** 2).mean(dim=(1, 2))
            acc = (mse_per_sample < Config.MSE_THRESHOLD).float().mean().item()
        return mse, acc
    
    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        
        if self.pruner.should_prune(epoch):
            pruned = self.pruner.prune(self.model)
            self.logger.info(f"Epoch {epoch}: Pruned {pruned:,} parameters")
        
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


        self.logger.info(f"Starting epoch: {self.start_epoch}, delta: {self.best_delta:.6f}")

        
        last_checkpoint_epoch = 0
        CHECKPOINT_INTERVAL = 500  # Cada 500 épocas antes de danger zone
        
        for epoch in range(1, MassiveLambdaConfig.MAX_EPOCHS + 1):
            train_metrics = self.train_epoch(epoch)
            val_mse, val_acc = self.validate()
            cryst_metrics = self.compute_discretization_metrics()
            
            current_delta = cryst_metrics['delta']
            current_lambda = self.criterion.lambda_quant
            
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
                'lambda_quant': current_lambda
            }
            self.metrics_history.append(record)
            
            # LOG cada 50 épocas como antes
            if epoch % 50 == 0:
                self.logger.info(
                    f"Epoch {self.start_epoch + epoch:4d} | "
                    f"Loss: {train_metrics['total_loss']:.6e} | "
                    f"MSE: {train_metrics['mse']:.6f} | "
                    f"Quant: {train_metrics['quant_penalty']:.4e} | "
                    f"ValAcc: {val_acc:.4f} | "
                    f"delta: {current_delta:.4f} -> {self.best_delta:.4f} | "
                    f"alpha: {cryst_metrics['alpha']:.2f} | "
                    f"Sparsity: {cryst_metrics['sparsity']:.2%} | "
                    f"Cryst: {cryst_metrics['weight_distribution']['crystallized']:.2%} | "
                    f"lambda: {current_lambda:.6e}"
                )
            
            # === ESTRATEGIA DE GUARDADO BRUTAL ===
            is_danger_zone = current_lambda >= 1e30
            should_save = False
            save_as_latest = False
            
            if is_danger_zone:
                # DANGER ZONE: guardar CADA ÉPOCA como latest (sobrescribir)
                should_save = True
                save_as_latest = True
            else:
                # Antes de danger zone: cada 500 épocas
                if epoch - last_checkpoint_epoch >= CHECKPOINT_INTERVAL:
                    should_save = True
                    save_as_latest = False
            
            # Guardar si corresponde
            if should_save:
                if save_as_latest:
                    # Sobrescribir latest.pth (rápido, no acumula archivos)
                    self._save_latest_checkpoint(epoch, cryst_metrics, val_acc)
                else:
                    # Guardar checkpoint con timestamp (cada 500 épocas)
                    self._save_crystal_checkpoint(epoch, cryst_metrics, val_acc, final=False)
                    last_checkpoint_epoch = epoch
            
            # === DETECCIÓN DE MEJORA (para seguimiento, no para guardar) ===
            improved = current_delta < self.best_delta * 0.99
            if improved:
                self.best_delta = current_delta
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # === CRECIENTO DE LAMBDA (sin lógica de guardado aquí) ===
            if self.epochs_without_improvement > CrystallizationConfig.PATIENCE:
                epochs_without = epoch - (epoch - self.epochs_without_improvement)
                self.logger.info(f"No improvement for {epochs_without} epochs. Growing lambda...")
                
                old_lambda = current_lambda
                
                if old_lambda < MassiveLambdaConfig.LAMBDA_MAX / MassiveLambdaConfig.LAMBDA_GROWTH:
                    new_lambda = old_lambda * MassiveLambdaConfig.LAMBDA_GROWTH
                else:
                    new_lambda = MassiveLambdaConfig.LAMBDA_MAX
                
                self.criterion.lambda_quant = new_lambda
                self.lambda_history.append(new_lambda)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.9
                
                self.pruner.prune(self.model, force_threshold=0.15)
                
                self.logger.info(f"New lambda: {old_lambda:.6e} -> {new_lambda:.6e}")
                self.epochs_without_improvement = 0
            
            # === NAN HANDLING: guardar como corrupted, NO salir ===
            # El latest.pth ya tiene el último válido, así que solo marcamos este como corrupto
            has_nan = (torch.isnan(torch.tensor(train_metrics['total_loss'])) or 
                    torch.isnan(torch.tensor(current_delta)))
            
            if has_nan:
                self.logger.error("=" * 60)
                self.logger.error(f"NaN DETECTED - Saving as corrupted checkpoint")
                self.logger.error(f"epoch: {self.start_epoch + epoch}")
                self.logger.error(f"LAST VALID CHECKPOINT: crystal_checkpoints/latest.pth")
                self.logger.error("=" * 60)
                
                self.logger.info("Continuing... (press Ctrl+C to stop)")
                # Opcional: podrías hacer break aquí si prefieres parar automáticamente
                # pero mejor dejar que el usuario decida
            
            # Verificar si llegamos al target (opcional, puedes quitarlo si quieres)
            if current_delta < MassiveLambdaConfig.TARGET_DELTA:
                self.logger.info("=" * 60)
                self.logger.info(f"TARGET REACHED! delta = {current_delta:.6f}")
                self.logger.info("=" * 60)
                self._save_crystal_checkpoint(epoch, cryst_metrics, val_acc, final=True)
                return self._compile_results(success=True, final_epoch=epoch)
            
            # Mantener accuracy si es necesario
            if (CrystallizationConfig.MAINTAIN_ACCURACY and 
                val_acc < 1.0 - CrystallizationConfig.ACCURACY_TOLERANCE):
                self.logger.warning(f"Accuracy dropped to {val_acc:.4f}. Restoring...")
                if self.best_state:
                    self.model.load_state_dict(self.best_state)
        
        # Fin del entrenamiento
        self.logger.info("=" * 60)
        self.logger.info(f"Completed. Best delta: {self.best_delta:.6f}")
        self.logger.info(f"Total epochs: {MassiveLambdaConfig.MAX_EPOCHS}")
        self.logger.info("=" * 60)
        return self._compile_results(success=False, final_epoch=MassiveLambdaConfig.MAX_EPOCHS)

    def _save_latest_checkpoint(self, epoch: int, metrics: Dict, val_acc: float):
        """Guarda/sobrescribe latest.pth - rápido, para danger zone"""
        os.makedirs("crystal_checkpoints", exist_ok=True)
        
        path = "crystal_checkpoints/latest.pth"
        
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
            },
            'timestamp': datetime.now().isoformat()
        }, path)
        
        # No loggear cada vez para no saturar, solo cada 10 guardados en danger zone
        if epoch % 2 == 0:
            self.logger.info(f"[DANGER ZONE] Latest checkpoint updated: epoch {self.start_epoch + epoch}")

        
    def _save_crystal_checkpoint(self, epoch: int, metrics: Dict, val_acc: float, 
                                final: bool = False, force_save: bool = False, 
                                emergency: bool = False):
        os.makedirs("crystal_checkpoints", exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Microsegundos para unicidad
        
        if emergency:
            suffix = f"EMERGENCY_epoch_{self.start_epoch + epoch}_delta_{metrics.get('delta', 'nan'):.4f}_{timestamp}"
        elif final:
            suffix = f"FINAL_{timestamp}"
        elif force_save:
            suffix = f"CRYSTAL100_epoch_{self.start_epoch + epoch}_delta_{metrics.get('delta', 0):.4f}_{timestamp}"
        else:
            suffix = f"epoch_{self.start_epoch + epoch}_delta_{metrics.get('delta', 0):.4f}_{timestamp}"
        
        path = f"crystal_checkpoints/crystal_{suffix}.pth"
        
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
        
        self.logger.info(f"Crystal checkpoint saved: {path}")
        
    def _compile_results(self, success: bool, final_epoch: int) -> Dict[str, Any]:
        initial_delta = self.initial_metrics.get('delta', 1.0)
        return {
            'success': success,
            'initial_delta': initial_delta,
            'final_delta': self.best_delta,
            'improvement_ratio': initial_delta / (self.best_delta + 1e-10),
            'total_epochs': final_epoch,
            'final_sparsity': self.pruner.get_sparsity(self.model),
            'lambda_history': self.lambda_history,
            'final_lambda': self.criterion.lambda_quant,
            'metrics_history': self.metrics_history,
            'timestamp': datetime.now().isoformat()
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Continue crystallization with massive lambda')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint (auto-search if not provided)')
    args = parser.parse_args()
    
    engine = ContinuationEngine(args.checkpoint)
    results = engine.refine()
    
    os.makedirs("results", exist_ok=True)
    results_path = f"results/continuation_massive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults: {results_path}")
    print(f"Success: {results['success']}")
    print(f"Improvement: {results['improvement_ratio']:.2f}x")
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Final lambda: {results['final_lambda']:.6e}")


if __name__ == "__main__":
    main()
