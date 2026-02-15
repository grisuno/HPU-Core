import torch
from safetensors.torch import save_file

# Cargamos el checkpoint
model_dict = torch.load("checkpoints/latest.pth", weights_only=False)

# Extraemos el diccionario de pesos real
if isinstance(model_dict, dict):
    # Buscamos la llave común en tu clase Trainer
    state_dict = model_dict.get('state_dict', model_dict.get('model_state_dict', model_dict))
else:
    state_dict = model_dict.state_dict()

# FILTRO PRO: Usamos torch.is_tensor para capturar Parameters y Tensors por igual
# Esto asegura que no guardes enteros como 'epoch' o estados del optimizador
tensors_only = {
    k: v.contiguous() if torch.is_tensor(v) else v.data.contiguous() 
    for k, v in state_dict.items() 
    if torch.is_tensor(v) or hasattr(v, 'data')
}

# Verificación de integridad para tu HPU-Core
if len(tensors_only) == 0:
    print("Error: No se detectaron tensores. Revisa la estructura del .pth")
else:
    save_file(tensors_only, "checkpoints/latest.safetensors")
    print(f"Exportación exitosa: {len(tensors_only)} tensores espectrales guardados.")
    # Imprime una llave para confirmar que los kernels están ahí
    print(f"Llaves detectadas: {list(tensors_only.keys())[:3]}...")