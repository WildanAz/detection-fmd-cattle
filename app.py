import os
from dotenv import load_dotenv
from app.config import UPLOAD_DIR
from app.routes import bp
from flask import Flask

load_dotenv()

app = Flask(__name__)

# Folder penyimpanan gambar sementara
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
