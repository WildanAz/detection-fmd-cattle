import os
from app.config import UPLOAD_DIR
from app.ml_service import deteksi_fitur, prediksi_klasifikasi, penanganan_dict
from flask import Blueprint, render_template, request, url_for
from werkzeug.utils import secure_filename

bp = Blueprint("routes", __name__)

# Routing utama
@bp.route("/", methods=["GET", "POST"])
def index():
    result = {}
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)
            image_url = url_for('static', filename=f'uploads/{filename}')

            with open(save_path, 'rb') as f:
                img_bytes = f.read()

            fitur, confidence = deteksi_fitur(img_bytes)

            warna = fitur['warna']
            tekstur = fitur['tekstur']
            lokasi = fitur['lokasi']
            luka = 'ya' if fitur['luka'] == 'luka' else 'tidak'

            hasil = prediksi_klasifikasi(warna, tekstur, lokasi, luka)
            rekomendasi = penanganan_dict[hasil]

            result = {
                "fitur": fitur,
                "confidence": confidence,
                "diagnosis": hasil,
                "rekomendasi": rekomendasi,
                "luka": luka
            }

            

    return render_template("index.html", result=result, image_url=image_url)