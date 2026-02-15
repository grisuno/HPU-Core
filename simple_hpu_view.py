import torch
import cv2
import numpy as np
import mss
import torch.nn.functional as F
from safetensors.torch import load_model
from experiment2 import HamiltonianNeuralNetwork

device = torch.device("cpu")
model = HamiltonianNeuralNetwork()

try:
    load_model(model, "checkpoints/latest.safetensors")
except Exception:
    exit()

model.to(device).eval()

sct = mss.mss()
monitor = sct.monitors[1]

while True:
    img = sct.grab(monitor)
    frame = np.array(img)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    h_orig, w_orig = gray.shape
    # El operador requiere gradientes para ver la dinámica del campo
    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        # 1. Proyección al espacio oculto del sistema
        # Esto genera el "potencial" inicial V(x)
        phi = F.gelu(model.input_proj(input_tensor))
        
        # 2. Aplicamos la evolución espectral (ĤΨ)
        # En lugar de irfft2 directo, buscamos la variación del campo
        layer = model.spectral_layers[0]
        
        x_fft = torch.fft.rfft2(phi)
        _, _, freq_h, freq_w = x_fft.shape
        
        kr = F.interpolate(layer.kernel_real.mean(dim=(0,1), keepdim=True), size=(freq_h, freq_w), mode='bilinear')
        ki = F.interpolate(layer.kernel_imag.mean(dim=(0,1), keepdim=True), size=(freq_h, freq_w), mode='bilinear')
        
        # Operación del kernel complejo sobre el espectro de la pantalla
        res_real = x_fft.real * kr - x_fft.imag * ki
        res_imag = x_fft.real * ki + x_fft.imag * kr
        
        # 3. Reconstrucción del Estado Evolucionado
        evolved_fft = torch.complex(res_real, res_imag)
        psi_t = torch.fft.irfft2(evolved_fft, s=(h_orig, w_orig))
        
        # 4. Visualización de la ACCIÓN (Hamiltoniana)
        # Lo que "ve" el modelo es la diferencia entre el estado original y el evolucionado
        # Esto resalta las singularidades donde el operador actúa con más fuerza
        action = torch.abs(psi_t.mean(dim=1) - phi.mean(dim=1)).squeeze()

    # Normalización local (Min-Max por frame)
    view = action.numpy()
    v_min, v_max = view.min(), view.max()
    
    if v_max > v_min:
        view = ((view - v_min) / (v_max - v_min + 1e-8) * 255).astype(np.uint8)
    else:
        view = np.zeros_like(view, dtype=np.uint8)
    
    # Usamos COLORMAP_JET para ver gradientes de presión (azul frío, rojo caliente)
    heatmap = cv2.applyColorMap(view, cv2.COLORMAP_JET)
    cv2.imshow("HPU-Core: Action Map", heatmap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()