from flask import Flask, request

app = Flask(__name__)

app.secret_key = "super secret key"
app.config['SESSION_TYPE'] = "filesystem"
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['avi'])

from app import views
