import os
import uuid
from flask import Flask, request, jsonify, render_template
from analyzer import analyze

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_route():
    if 'video' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

    file = request.files['video']

    if not file.filename:
        return jsonify({'error': 'Nome de arquivo vazio.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Formato não suportado. Use MP4, AVI, MOV, MKV ou WEBM.'}), 400

    ext      = os.path.splitext(file.filename)[1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4()}{ext}')
    file.save(tmp_path)

    try:
        result = analyze(tmp_path)
    finally:
        try:
            os.remove(tmp_path)  # apaga imediatamente após análise (LGPD)
        except OSError:
            pass

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
