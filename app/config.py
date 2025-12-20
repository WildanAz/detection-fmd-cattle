from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent

MODEL_DIR = BASE_DIR / "models"
SVM_PATH = MODEL_DIR / "SVM_linear.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

UPLOAD_DIR = BASE_DIR / "static" / "uploads"

AZURE_PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

PROJECT_ID = "fbdee0e8-0a8f-4269-91ba-942e968805d0"

URLS = {
    "lokasi": f"{AZURE_ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/Iteration3/image",
    "warna": f"{AZURE_ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/Iteration7/image",
    "tekstur": f"{AZURE_ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/Iteration6/image",
    "luka": f"{AZURE_ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/classify/iterations/Iteration4/image"
}

WARNA_KAT  = ['hitam', 'kuning', 'merah']
TEKSTUR_KAT = ['halus', 'kasar']
LOKASI_KAT  = ['gusi', 'kuku', 'lidah']
CLASS_LABELS = ['Foot_rot', 'necrotic_stomatitis', 'pmk', 'sehat']

PENANGANAN_DICT = {
    'pmk': [
        "Isolasi sapi yang terinfeksi untuk mencegah penyebaran lebih lanjut.",
        "Pemberian antibiotik untuk mencegah infeksi sekunder serta analgesik untuk mengurangi rasa sakit.",
        "Vaksinasi PMK sebagai langkah pencegahan utama, terutama di wilayah endemik."
    ],
    'Foot_rot': [
        "Pembersihan luka dan aplikasi antibiotik topikal untuk menghambat pertumbuhan bakteri.",
        "Pemberian antibiotik sistemik, dalam kasus infeksi yang lebih parah.",
        "Peningkatan sanitasi kandang, terutama menjaga lantai tetap kering dan bersih."
    ],
    'necrotic_stomatitis': [
        "Pemberian antibiotik sistemik untuk menghambat pertumbuhan bakteri.",
        "Perawatan luka dengan antiseptik oral untuk mempercepat penyembuhan.",
        "Peningkatan sanitasi pakan dan air minum untuk mencegah infeksi ulang."
    ],
    'sehat': ["Tidak perlu penanganan khusus."]
}