from flask import Flask, request, jsonify
import os
from face_utils import save_user_face, match_face
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'static/faces'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return jsonify({"message": "Face Recognition API is running."})

@app.route('/api/register', methods=['POST'])
def api_register():
    userid = request.form.get('userid')
    file = request.files.get('image')

    if not userid or not file:
        return jsonify({"error": "User ID and image are required."}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)

    success, message = save_user_face(temp_path, userid)
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 400


@app.route('/api/match', methods=['POST'])
def api_match():
    print("Incoming POST request to /api/match")

    print("Form keys:", request.form.keys())
    print("File keys:", request.files.keys())

    userid = request.form.get('userid')
    file = request.files.get('image')

    print("userid =", userid)
    print("file =", file)
    print("file.filename =", file.filename if file else "No file")

    if not userid or not file:
        return jsonify({"error": "User ID and image are required."}), 400

    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)

    is_match = match_face(userid, temp_path)
    bool_response = is_match[0]
    return jsonify({"match": bool_response}), 200

if __name__ == '__main__':
    app.run(debug=True)
