import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "DB_NAME": "RealSecurityDB",
    "USE_CUDA": False,
    "IMAGE_FOLDER": "data/img",
    "FACES_FOLDER": "data/data_faces",
    "MTCNN_PATH": "utils/mtcnn.py",
    "CREATE_PATH": "setup_db.sql",
    "ENTRANCE_CAMERA_INDEX": 0,
    "EXIT_CAMERA_INDEX": 1,
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 480,
    "open_count": 0,
    "SIMILARITY_THRESHOLD": 0.85,
    "MTCNN_THRESHOLDS": [0.6, 0.7, 0.7],
    "MTCNN_MIN_FACE_SIZE": 40,
    "HAAR_SCALE_FACTOR": 1.1,
    "HAAR_MIN_NEIGHBORS": 5,
    "HAAR_MIN_SIZE": [30, 30],
    "PREPROCESS_RESIZE": [160, 160],
    "DETECTION_CONFIDENCE": 0.6,
    "TIME_RECOGNITION": 3,
    "STATE_APP": True,
    "NOTIFICATION_ASISTENCIA": True,
    "NOTIFICACION_INTRUSO": True,
    "PHONE_NUMBER": "",
    "MAIL_USER": "",

    "ENTRANCE_CAMERA_MODE": "local",
    "ENTRANCE_CAMERA_URL": "",
    "EXIT_CAMERA_MODE": "local",
    "EXIT_CAMERA_URL": "",

    "REPORT_PERIOD": "Mensual",   # Diario, Semanal, Mensual, Anual
    "ENABLE_EMAIL_REPORTS": False,
    "REPORT_EMAIL": "",
    
    "ENTRY_INTERVAL_START": "",
    "ENTRY_INTERVAL_END": "",
    "EXIT_INTERVAL_START": "",
    "EXIT_INTERVAL_END": "",
    
    # Nuevo campo para guardar la fecha del último reporte exitoso
    "LAST_REPORT_DATE": ""  # se guardará como string 'YYYY-MM-DD', por ejemplo
}

def initialize_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)

    faces_folder = DEFAULT_CONFIG["FACES_FOLDER"]
    if not os.path.exists(faces_folder):
        try:
            os.makedirs(faces_folder)
            print(f"Directorio creado: {faces_folder}")
        except Exception as e:
            print(f"Error al crear el directorio {faces_folder}: {e}")

def load_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Archivo de configuración no encontrado. Usando valores predeterminados.")
        return DEFAULT_CONFIG
    except json.JSONDecodeError:
        print("Error al leer el archivo de configuración. Usando valores predeterminados.")
        return DEFAULT_CONFIG

def save_config(config):
    try:
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file, indent=4)
        print("Configuración guardada correctamente.")
    except Exception as e:
        print(f"Error al guardar el archivo de configuración: {e}")

def get_config_value(key, default=None):
    config = load_config()
    return config.get(key, default)

def set_config_value(key, value):
    config = load_config()
    config[key] = value
    save_config(config)

def get_device():
    use_cuda = get_config_value("USE_CUDA", True)
    if use_cuda and torch.cuda.is_available():
        print("Usando GPU: CUDA")
        return torch.device('cuda')
    else:
        print("Usando CPU")
        return torch.device('cpu')

# Inicializamos el config si no existe
initialize_config()

# Variables cargadas desde la configuración
DB_NAME = get_config_value("DB_NAME", "default.db")
USE_CUDA = get_config_value("USE_CUDA", False)
CASCADE_PATH = get_config_value("CASCADE_PATH", "default.xml")
CREATE_PATH = get_config_value("CREATE_PATH", "default.sql")
MTCNN_PATH = get_config_value("MTCNN_PATH", "default_mtcnn.py")
FRAME_WIDTH = get_config_value("FRAME_WIDTH", 640)
FRAME_HEIGHT = get_config_value("FRAME_HEIGHT", 480)
FACES_FOLDER = get_config_value("FACES_FOLDER")
STATE_APP = get_config_value("STATE_APP", True)
ENTRANCE_CAMERA_INDEX = get_config_value("ENTRANCE_CAMERA_INDEX", 0)
EXIT_CAMERA_INDEX = get_config_value("EXIT_CAMERA_INDEX", 1)
PREPROCESS_RESIZE = get_config_value("PREPROCESS_RESIZE", [160, 160])
HAAR_SCALE_FACTOR = get_config_value("HAAR_SCALE_FACTOR", 1.1)
HAAR_MIN_NEIGHBORS = get_config_value("HAAR_MIN_NEIGHBORS", 5)
HAAR_MIN_SIZE = get_config_value("HAAR_MIN_SIZE", [30, 30])
SIMILARITY_THRESHOLD = get_config_value("SIMILARITY_THRESHOLD", 0.85)
MTCNN_THRESHOLDS = get_config_value("MTCNN_THRESHOLDS", [0.6, 0.7, 0.7])
MTCNN_MIN_FACE_SIZE = get_config_value("MTCNN_MIN_FACE_SIZE", 40)
DETECTION_CONFIDENCE = get_config_value("DETECTION_CONFIDENCE", 0.6)
TIME_RECOGNITION = get_config_value("TIME_RECOGNITION", 3)
NOTIFICATION_ASISTENCIA = get_config_value("NOTIFICATION_ASISTENCIA", True)
NOTIFICACION_INTRUSO = get_config_value("NOTIFICACION_INTRUSO", True)
PHONE_NUMBER = get_config_value("PHONE_NUMBER", "")
EMAIL_USER = get_config_value("EMAIL_USER", "")
REPORT_PERIOD = get_config_value("REPORT_PERIOD", "Mensual")
ENABLE_EMAIL_REPORTS = get_config_value("ENABLE_EMAIL_REPORTS", False)
REPORT_EMAIL = get_config_value("REPORT_EMAIL", "")

ENTRANCE_CAMERA_MODE = get_config_value("ENTRANCE_CAMERA_MODE", "local")
ENTRANCE_CAMERA_URL  = get_config_value("ENTRANCE_CAMERA_URL", "")
EXIT_CAMERA_MODE     = get_config_value("EXIT_CAMERA_MODE", "local")
EXIT_CAMERA_URL      = get_config_value("EXIT_CAMERA_URL", "")

ENTRY_INTERVAL_START = get_config_value("ENTRY_INTERVAL_START", "")
ENTRY_INTERVAL_END   = get_config_value("ENTRY_INTERVAL_END", "")
EXIT_INTERVAL_START  = get_config_value("EXIT_INTERVAL_START", "")
EXIT_INTERVAL_END    = get_config_value("EXIT_INTERVAL_END", "")

# Nuevo: Guardamos la última fecha en que se generó un reporte
LAST_REPORT_DATE = get_config_value("LAST_REPORT_DATE", "")
