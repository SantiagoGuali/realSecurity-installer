import onnx
from onnxruntime.tools import optimize_model

# Ruta del modelo original y optimizado
original_model_path = "C:/Users/Santiago.G/.insightface/models/buffalo_l/w600k_r50.onnx"
optimized_model_path = "C:/Users/Santiago.G/.insightface/models/buffalo_l/w600k_r50_optimized.onnx"

# Cargar modelo ONNX
model = onnx.load(original_model_path)

# Aplicar optimización
optimized_model = optimize_model(model)

# Guardar el modelo optimizado
onnx.save(optimized_model, optimized_model_path)
print("✅ Modelo ONNX optimizado guardado correctamente.")

