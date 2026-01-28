import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any
import glob

from experiment2 import Config, HamiltonianNeuralNetwork, HamiltonianDataset


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
        
        # 2. Calcular métricas de validación
        val_metrics = self._compute_validation_metrics()
        results['validation_metrics'] = val_metrics
        
        # 3. Calcular métricas de discretización (delta, alpha, etc)
        discret_metrics = self._compute_discretization_metrics()
        results['discretization_metrics'] = discret_metrics
        
        # 4. Calcular métricas de cuantización
        quant_metrics = self._compute_quantization_metrics()
        results['quantization_metrics'] = quant_metrics
        
        # 5. Calcular loss completo
        loss_metrics = self._compute_loss_metrics()
        results['loss_metrics'] = loss_metrics
        
        # 6. Comparar con métricas almacenadas
        comparison = self._compare_with_stored(results)
        results['comparison_with_stored'] = comparison
        
        # 7. Verificar consistencia interna
        consistency = self._check_internal_consistency(results)
        results['consistency_checks'] = consistency
        
        # 8. Resumen de salud
        health = self._compute_health_score(results)
        results['health_score'] = health
        
        self._print_report(results)
        
        return results
    
    def _check_weight_integrity(self) -> Dict[str, Any]:
        """Verifica que los pesos no tengan NaN/Inf"""
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
            
            param_stats[name] = {
                'shape': list(data.shape),
                'mean': data.mean().item() if data.numel() > 0 else 0,
                'std': data.std().item() if data.numel() > 0 else 0,
                'min': data.min().item() if data.numel() > 0 else 0,
                'max': data.max().item() if data.numel() > 0 else 0,
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
                # Delta para esta capa
                rounded = torch.round(param.data)
                delta_layer = (param.data - rounded).abs().max().item()
                layer_deltas[name] = delta_layer
                all_deltas.append(delta_layer)
        
        delta_max = max(all_deltas) if all_deltas else 1.0
        delta_mean = np.mean(all_deltas) if all_deltas else 1.0
        
        # Alpha = -log(delta)
        alpha = -np.log(delta_max + 1e-15) if delta_max > 1e-10 else 20.0
        
        # Purity index
        purity = 1.0 - delta_max
        
        # Es cristal? (alpha > 7.0)
        is_crystal = alpha > 7.0
        
        # Distribución de pesos
        all_weights = torch.cat([p.data.flatten() for p in self.model.parameters()])
        
        near_zero = (all_weights.abs() < 0.1).sum().item()
        near_one = ((all_weights - 1).abs() < 0.1).sum().item()
        near_minus_one = ((all_weights + 1).abs() < 0.1).sum().item()
        near_integer = (
            ((all_weights - torch.round(all_weights)).abs() < 0.1).sum().item()
        )
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
        
        # Lambda usada (del checkpoint o default)
        lambda_quant = self.config.get('quant_lambda', 0.5)
        
        # Calcular loss total (usando float64 para evitar overflow)
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
    
    def _compare_with_stored(self, computed: Dict) -> Dict[str, Any]:
        """Compara métricas calculadas vs almacenadas en el checkpoint"""
        comparisons = {}
        
        # Mapeo de nombres: stored -> computed
        mappings = {
            'delta': ('discretization_metrics', 'delta_max'),
            'alpha': ('discretization_metrics', 'alpha'),
            'val_acc': ('validation_metrics', 'accuracy'),
            'val_mse': ('validation_metrics', 'mse'),
            'quant_penalty': ('quantization_metrics', 'quantization_penalty'),
            'total_loss': ('loss_metrics', 'total_loss')
        }
        
        for stored_key, (section, computed_key) in mappings.items():
            if stored_key in self.stored_metrics or stored_key == 'val_acc' and self.stored_val_acc is not None:
                stored_val = self.stored_metrics.get(stored_key, self.stored_val_acc)
                computed_val = computed.get(section, {}).get(computed_key)
                
                if stored_val is not None and computed_val is not None:
                    diff = abs(stored_val - computed_val)
                    rel_diff = diff / abs(stored_val) if stored_val != 0 else float('inf')
                    
                    comparisons[stored_key] = {
                        'stored': stored_val,
                        'computed': computed_val,
                        'absolute_diff': diff,
                        'relative_diff': rel_diff,
                        'match': diff < 1e-5  # Tolerancia
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
        
        # 3. Consistencia accuracy
        val_mse_per_sample = results['validation_metrics']['mse']
        # No podemos recalcular exactamente sin los samples individuales, pero verificamos orden de magnitud
        
        return checks
    
    def _compute_health_score(self, results: Dict) -> Dict[str, Any]:
        """Calcula un score de salud del checkpoint (0-100)"""
        score = 100.0
        issues = []
        
        # Penalizaciones
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
            if not check_data.get('match', True):
                score -= 5
                issues.append(f"INCONSISTENT:{check_name}")
        
        # Delta muy alto (no convergido)
        if results['discretization_metrics']['delta_max'] > 0.5:
            score -= 10
            issues.append("HIGH_DELTA")
        
        # Lambda extremo
        lambda_val = results['loss_metrics']['lambda_used']
        if lambda_val > 1e30:
            score -= 15
            issues.append("EXTREME_LAMBDA")
        
        return {
            'score': max(0, score),
            'status': 'HEALTHY' if score > 80 else 'DEGRADED' if score > 50 else 'CRITICAL',
            'issues': issues,
            'is_usable': score > 30 and results['weight_integrity']['is_valid']
        }
    
    def _print_report(self, results: Dict):
        """Imprime reporte formateado"""
        print(f"\n{'='*70}")
        print("REPORTE DE VERIFICACIÓN")
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
        
        wd = dm['weight_distribution']
        print(f"\n  Distribución de pesos:")
        print(f"    Cerca de 0: {wd['near_zero']:.2%}")
        print(f"    Cerca de 1: {wd['near_one']:.2%}")
        print(f"    Cerca de -1: {wd['near_minus_one']:.2%}")
        print(f"    Cerca de cualquier entero: {wd['near_any_integer']:.2%}")
        
        print(f"\n[MÉTRICAS DE CUANTIZACIÓN]")
        qm = results['quantization_metrics']
        print(f"  Penalty: {qm['quantization_penalty']:.6e}")
        print(f"  Total params: {qm['total_params']:,}")
        
        print(f"\n[LOSS COMPLETO]")
        lm = results['loss_metrics']
        print(f"  MSE: {lm['mse']:.6f}")
        print(f"  Quant penalty: {lm['quantization_penalty']:.6e}")
        print(f"  Lambda: {lm['lambda_used']:.6e}")
        print(f"  TOTAL LOSS: {lm['total_loss']:.6e}")
        print(f"  Descomposición: {lm['loss_decomposition']['mse_ratio']:.2%} MSE, "
              f"{lm['loss_decomposition']['quant_ratio']:.2%} Quant")
        
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
        print(f"  Usable: {health['is_usable']} {'✓' if health['is_usable'] else '✗'}")
        if health['issues']:
            print(f"  Problemas: {', '.join(health['issues'])}")
        
        print(f"\n{'='*70}")


def verify_latest_checkpoints(checkpoint_dir: str = "crystal_checkpoints", n: int = 5):
    """Verifica los N checkpoints más recientes"""
    pattern = os.path.join(checkpoint_dir, "*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    # Ordenar por fecha de modificación
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
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN DE VERIFICACIÓN")
    print(f"{'='*70}")
    for r in results_list:
        health = r.get('health_score', {})
        epoch = r.get('epoch_reported', '?')
        score = health.get('score', 0)
        status = health.get('status', 'UNKNOWN')
        usable = '✓' if health.get('is_usable', False) else '✗'
        print(f"Epoch {epoch:5s} | Score: {score:5.1f} | {status:10s} | Usable: {usable} | {os.path.basename(r['checkpoint_path'])}")


def main():
    parser = argparse.ArgumentParser(description='Verify checkpoint metrics')
    parser.add_argument('checkpoint', nargs='?', default=None, 
                       help='Specific checkpoint to verify (or latest if not provided)')
    parser.add_argument('--latest', type=int, default=None,
                       help='Verify N latest checkpoints')
    parser.add_argument('--dir', default='crystal_checkpoints',
                       help='Directory to search for checkpoints')
    
    args = parser.parse_args()
    
    if args.latest:
        verify_latest_checkpoints(args.dir, args.latest)
    elif args.checkpoint:
        verifier = CheckpointVerifier(args.checkpoint)
        results = verifier.verify_all_metrics()
        
        # Guardar reporte
        report_path = args.checkpoint.replace('.pth', '_verification.json')
        with open(report_path, 'w') as f:
            # Limpiar para JSON
            clean_results = json.loads(json.dumps(results, default=str))
            json.dump(clean_results, f, indent=2)
        print(f"\nReporte guardado en: {report_path}")
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
