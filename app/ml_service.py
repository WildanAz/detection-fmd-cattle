import numpy as np
import joblib
import requests
from app.config import AZURE_PREDICTION_KEY, SVM_PATH, URLS, SCALER_PATH, PENANGANAN_DICT
from app.config import WARNA_KAT, TEKSTUR_KAT, LOKASI_KAT, CLASS_LABELS

# Load model dan scaler
model = joblib.load(SVM_PATH)
scaler = joblib.load(SCALER_PATH)

# Kategori fitur
warna_kategori = WARNA_KAT
tekstur_kategori = TEKSTUR_KAT
lokasi_kategori = LOKASI_KAT
class_labels = CLASS_LABELS

# Saran penanganan
penanganan_dict = PENANGANAN_DICT

# Azure Prediction
headers = {
    'Content-Type': 'application/octet-stream',
    'Prediction-Key': AZURE_PREDICTION_KEY
}

# Fungsi bantu
def one_hot_encode(value, categories):
    one_hot = [0] * len(categories)
    if value in categories:
        one_hot[categories.index(value)] = 1
    return one_hot

def deteksi_fitur(img_bytes):
    hasil, confidence = {}, {}
    for fitur in URLS:
        res = requests.post(URLS[fitur], headers=headers, data=img_bytes)
        pred = res.json()["predictions"]
        top = max(pred, key=lambda x: x["probability"])
        hasil[fitur] = top["tagName"].lower().replace("_", "")
        confidence[fitur] = {
            p["tagName"].lower().replace("_", ""): round(p["probability"] * 100, 2)
            for p in pred
        }
    return hasil, confidence

def prediksi_klasifikasi(warna, tekstur, lokasi, luka):
    warna_oh = one_hot_encode(warna, warna_kategori)
    tekstur_oh = one_hot_encode(tekstur, tekstur_kategori)
    lokasi_oh = one_hot_encode(lokasi, lokasi_kategori)
    luka_bin = 1 if luka == 'ya' else 0

    input_data = np.array([[luka_bin] + warna_oh + tekstur_oh + lokasi_oh])
    input_scaled = scaler.transform(input_data)
    probas = model.predict_proba(input_scaled)[0]
    pred_idx = np.argmax(probas)
    pred_label = class_labels[pred_idx]

    if luka_bin == 1 and pred_label == 'sehat':
        for i in np.argsort(probas)[::-1]:
            if class_labels[i] != 'sehat':
                pred_label = class_labels[i]
                break
    if luka_bin == 0 and pred_label != 'sehat':
        pred_label = 'sehat'

    if pred_label == 'necrotic_stomatitis' and (warna == 'merah' or lokasi == 'kuku'):
        for i in np.argsort(probas)[::-1]:
            if class_labels[i] != 'necrotic_stomatitis':
                pred_label = class_labels[i]
                break
    return pred_label
