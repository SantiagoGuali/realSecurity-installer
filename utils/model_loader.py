import os
import tensorflow as tf

# 📌 Ruta de caché para el modelo
CACHE_PATH = "cache/facenet_model.pb"
FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

def load_facenet_model(model_path):
    """Carga FaceNet y lo almacena en caché usando `tf.io.write_graph()`."""
    
    if os.path.exists(CACHE_PATH):  # 📌 Si el modelo ya está cacheado, cárgalo
        print("✅ Cargando modelo FaceNet desde caché...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.io.gfile.GFile(CACHE_PATH, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        return graph

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ El modelo FaceNet no se encontró en: {model_path}")

    try:
        print("✅ Cargando modelo FaceNet desde archivo...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

        # 📌 Guardar en caché usando `tf.io.write_graph`
        os.makedirs("cache", exist_ok=True)
        tf.io.write_graph(graph.as_graph_def(), "cache", "facenet_model.pb", as_text=False)
        print("✅ Modelo FaceNet guardado en caché.")

    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar FaceNet: {e}")

    return graph

# 📌 Definir `graph` globalmente después de cargar el modelo
graph = load_facenet_model(FACENET_MODEL_PATH)
sess = tf.compat.v1.Session(graph=graph)  # 📌 Mantener sesión abierta
