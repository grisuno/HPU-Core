import torch
import cv2
import numpy as np
import mss
import torch.nn.functional as F
from safetensors.torch import load_model
from experiment2 import HamiltonianNeuralNetwork

# Configuración del dispositivo
device = torch.device("cpu")
model = HamiltonianNeuralNetwork()

# Carga del modelo
try:
    load_model(model, "checkpoints/latest.safetensors")
    print("Modelo HPU-Core cargado exitosamente.")
except Exception as e:
    print(f"Error crítico cargando el modelo: {e}")
    exit()

model.to(device).eval()

sct = mss.mss()
monitor = sct.monitors[1]

print("Iniciando visualización Hamiltoniana Integrada...")
print("Ventana 1: Energy Density (Resonancia Interna)")
print("Ventana 2: Topological Phase (Vórtices)")
print("Ventana 3: Action Map (Visión Nítida)")
print("Presiona 'q' para salir.")

while True:
    # 1. Captura de Pantalla
    img = sct.grab(monitor)
    frame = np.array(img)
    
    # Preprocesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    h_orig, w_orig = gray.shape
    
    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        # Proyección al Espacio Oculto (Estado Inicial)
        phi = F.gelu(model.input_proj(input_tensor))
        
        layer = model.spectral_layers[0]
        
        # ---------------------------------------------------------
        # VISIÓN 1 & 2: DOMINIO COMPLEJO (Topología y Resonancia)
        # ---------------------------------------------------------
        # Usamos FFT completa para ver la estructura interna del campo
        x_fft_complex = torch.fft.fft2(phi)
        _, _, freq_h_c, freq_w_c = x_fft_complex.shape
        
        # Interpolamos kernel para tamaño completo
        kr_c = F.interpolate(layer.kernel_real.mean(dim=(0,1), keepdim=True), size=(freq_h_c, freq_w_c), mode='bilinear')
        ki_c = F.interpolate(layer.kernel_imag.mean(dim=(0,1), keepdim=True), size=(freq_h_c, freq_w_c), mode='bilinear')
        
        # Evolución compleja
        res_real_c = x_fft_complex.real * kr_c - x_fft_complex.imag * ki_c
        res_imag_c = x_fft_complex.real * ki_c + x_fft_complex.imag * kr_c
        evolved_fft_complex = torch.complex(res_real_c, res_imag_c)
        
        # Reconstrucción compleja
        psi_t_complex = torch.fft.ifft2(evolved_fft_complex, s=(h_orig, w_orig))
        
        # Mapas
        amplitude_map = torch.abs(psi_t_complex).mean(dim=1).squeeze()
        phase_map = torch.angle(psi_t_complex).mean(dim=1).squeeze()

        # ---------------------------------------------------------
        # VISIÓN 3: ACTION MAP (Lo que "ve" el modelo claramente)
        # ---------------------------------------------------------
        # Usamos RFFT (Real) como en tu segundo script, enfocado en la acción
        x_fft_real = torch.fft.rfft2(phi)
        _, _, freq_h_r, freq_w_r = x_fft_real.shape
        
        # Interpolamos kernel para tamaño real (rfft tiene ancho reducido)
        kr_r = F.interpolate(layer.kernel_real.mean(dim=(0,1), keepdim=True), size=(freq_h_r, freq_w_r), mode='bilinear')
        ki_r = F.interpolate(layer.kernel_imag.mean(dim=(0,1), keepdim=True), size=(freq_h_r, freq_w_r), mode='bilinear')
        
        # Evolución real
        res_real_r = x_fft_real.real * kr_r - x_fft_real.imag * ki_r
        res_imag_r = x_fft_real.real * ki_r + x_fft_real.imag * kr_r
        
        evolved_fft_real = torch.complex(res_real_r, res_imag_r)
        psi_t_real = torch.fft.irfft2(evolved_fft_real, s=(h_orig, w_orig))
        
        # Cálculo de la ACCIÓN (Diferencia absoluta)
        action = torch.abs(psi_t_real.mean(dim=1) - phi.mean(dim=1)).squeeze()

    # --- RENDERIZADO DE LAS 3 VENTANAS ---

    # 1. Amplitud (Resonancia)
    amp_np = amplitude_map.numpy()
    v_min, v_max = amp_np.min(), amp_np.max()
    if v_max > v_min:
        amp_view = ((amp_np - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    else:
        amp_view = np.zeros_like(amp_np, dtype=np.uint8)
    heatmap_amp = cv2.applyColorMap(amp_view, cv2.COLORMAP_JET)

    # 2. Fase (Topología)
    phase_np = phase_map.numpy()
    phase_norm = ((phase_np + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    heatmap_phase = cv2.applyColorMap(phase_norm, cv2.COLORMAP_TWILIGHT)

    # 3. Acción (Visión Nítida)
    act_np = action.numpy()
    v_min_a, v_max_a = act_np.min(), act_np.max()
    if v_max_a > v_min_a:
        act_view = ((act_np - v_min_a) / (v_max_a - v_min_a + 1e-8) * 255).astype(np.uint8)
    else:
        act_view = np.zeros_like(act_np, dtype=np.uint8)
    heatmap_act = cv2.applyColorMap(act_view, cv2.COLORMAP_JET)
    
    # Mostrar
    cv2.imshow("1. Energy Density (Resonancia)", heatmap_amp)
    cv2.imshow("2. Topological Phase (Vortices)", heatmap_phase)
    cv2.imshow("3. Action Map (Vision Clara)", heatmap_act)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()