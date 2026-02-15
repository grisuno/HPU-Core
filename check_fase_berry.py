import torch
import cv2
import numpy as np
import mss
import torch.nn.functional as F
from safetensors.torch import load_model
from experiment2 import HamiltonianNeuralNetwork  # Asegúrate de que está en el path

# Configuración
device = torch.device("cpu")
model = HamiltonianNeuralNetwork()
try:
    load_model(model, "checkpoints/latest.safetensors")
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()
model.to(device).eval()

sct = mss.mss()
monitor = sct.monitors[1]

print("Midiendo invariantes topológicos en tiempo real...")
print("Ventana 1: Action Map (diferencia entrada-evolución)")
print("Ventana 2: Phase Map (topología)")
print("Presiona 'q' para salir.")

while True:
    # Captura de pantalla
    img = sct.grab(monitor)
    frame = np.array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    h_orig, w_orig = gray.shape
    input_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        # Proyección al espacio oculto
        phi = F.gelu(model.input_proj(input_tensor))  # [1, C, H, W]
        
        # Tomamos la primera capa espectral (como en tu código)
        layer = model.spectral_layers[0]

        # ------------------------------------------------------------
        # 1. Cálculo del Action Map (usando rfft, como en tu versión rápida)
        # ------------------------------------------------------------
        x_fft_r = torch.fft.rfft2(phi)
        _, _, freq_h_r, freq_w_r = x_fft_r.shape
        kr_r = F.interpolate(layer.kernel_real.mean(dim=(0,1), keepdim=True), size=(freq_h_r, freq_w_r), mode='bilinear')
        ki_r = F.interpolate(layer.kernel_imag.mean(dim=(0,1), keepdim=True), size=(freq_h_r, freq_w_r), mode='bilinear')
        res_real_r = x_fft_r.real * kr_r - x_fft_r.imag * ki_r
        res_imag_r = x_fft_r.real * ki_r + x_fft_r.imag * kr_r
        evolved_fft_r = torch.complex(res_real_r, res_imag_r)
        psi_t_r = torch.fft.irfft2(evolved_fft_r, s=(h_orig, w_orig))
        action = torch.abs(psi_t_r.mean(dim=1) - phi.mean(dim=1)).squeeze().cpu().numpy()

        # ------------------------------------------------------------
        # 2. Mapa de fase (usando FFT completa)
        # ------------------------------------------------------------
        x_fft_c = torch.fft.fft2(phi)
        _, _, freq_h_c, freq_w_c = x_fft_c.shape
        kr_c = F.interpolate(layer.kernel_real.mean(dim=(0,1), keepdim=True), size=(freq_h_c, freq_w_c), mode='bilinear')
        ki_c = F.interpolate(layer.kernel_imag.mean(dim=(0,1), keepdim=True), size=(freq_h_c, freq_w_c), mode='bilinear')
        res_real_c = x_fft_c.real * kr_c - x_fft_c.imag * ki_c
        res_imag_c = x_fft_c.real * ki_c + x_fft_c.imag * kr_c
        evolved_fft_c = torch.complex(res_real_c, res_imag_c)
        psi_t_c = torch.fft.ifft2(evolved_fft_c, s=(h_orig, w_orig))
        phase_map = torch.angle(psi_t_c).mean(dim=1).squeeze().cpu().numpy()  # [H, W]

        # ------------------------------------------------------------
        # 3. Métricas topológicas sobre el mapa de fase
        # ------------------------------------------------------------
        # a) Fase media y desviación
        mean_phase = np.mean(phase_map)
        std_phase = np.std(phase_map)

        # b) Número de enrollamiento aproximado (circulación en el borde)
        # Tomamos los píxeles del borde: superior, inferior, izquierdo, derecho
        # Calculamos la diferencia de fase a lo largo del perímetro (con desenrollado simple)
        h, w = phase_map.shape
        # Borde superior (fila 0, columnas 0..w-1)
        top = phase_map[0, :]
        # Borde inferior (fila h-1, columnas w-1..0 invertido para mantener orientación)
        bottom = phase_map[-1, ::-1]
        # Borde derecho (columna w-1, filas 1..h-2) (evitamos esquinas repetidas)
        right = phase_map[1:-1, -1]
        # Borde izquierdo (columna 0, filas h-2..1 invertido)
        left = phase_map[-2:0:-1, 0]
        # Unimos en orden antihorario empezando en (0,0)
        boundary = np.concatenate([top, right, bottom, left])
        # Desenrollamos para evitar saltos de 2π
        boundary_unwrapped = np.unwrap(boundary)
        # La circulación total es la diferencia entre el último y el primer valor
        circulation = boundary_unwrapped[-1] - boundary_unwrapped[0]
        winding_number = circulation / (2 * np.pi)  # debería ser cercano a entero

        # Mostramos las métricas en consola
        print(f"\rFase media: {mean_phase:6.3f} rad | Std: {std_phase:6.3f} | Winding: {winding_number:6.3f}", end="")

        # ------------------------------------------------------------
        # Visualización
        # ------------------------------------------------------------
        # Action Map
        v_min, v_max = action.min(), action.max()
        if v_max > v_min:
            action_view = ((action - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            action_view = np.zeros_like(action, dtype=np.uint8)
        action_heat = cv2.applyColorMap(action_view, cv2.COLORMAP_JET)

        # Phase Map (normalizado a [0,255] para visualización)
        phase_norm = ((phase_map + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        phase_heat = cv2.applyColorMap(phase_norm, cv2.COLORMAP_TWILIGHT)

        cv2.imshow("Action Map (Hamiltoniano)", action_heat)
        cv2.imshow("Phase Map", phase_heat)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()