import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils.settings_controller import SIMILARITY_THRESHOLD   
def ajustar_umbral(similitudes, margen=5):
    if not similitudes:
        return SIMILARITY_THRESHOLD

    # 🔹 Obtener los valores de similitud correctos
    valores_similitud = np.array(similitudes)
    
    # 🔹 Determinar el percentil 10 para ajustar el umbral
    nuevo_umbral = np.percentile(valores_similitud, 10) - (margen / 100)

    return max(nuevo_umbral, 0.4)  # 🔹 No permitir umbrales demasiado bajos
