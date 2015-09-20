from flask import Flask
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug import secure_filename
import json
import uuid

UPLOAD_FOLDER = '/uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'wav', 'blob', 'bin'])

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return "Hello World!"

SPEAKER_DATA_PATH = 'data/speakers.json'

@app.route('/add_speaker_form/', methods=['GET', 'POST'])
def add_speaker():
    if request.form:
        fullname = request.form.get('fullname',"")
        email = request.form.get('email',"")
        gender = request.form.get('gender', "")

        filename = request.form.get('file', None)
        audio_file = request.files.get('file', None) #fails here
        rand_file_name_ending = str(uuid.uuid4())

        new_file_name = "user_files/" + rand_file_name_ending + ".wav"
        audio_file.save(new_file_name)

        data = {
        "fullname":fullname,
        "email": email,
        "gender":gender,
        "filename": new_file_name}
        save_speaker(data)

    return render_template('add_speaker_form.html')

@app.route('/add_meeting/', methods=['GET', 'POST'])
def add_meeting():
    if request.form:
        title = request.form.get('title',"")
        filename = request.form.get('file', None)
        audio_file = request.files.get('file', None) #fails here
        rand_file_name_ending = str(uuid.uuid4())
        new_file_name = "meeting_files/" + rand_file_name_ending + ".wav"
        audio_file.save(new_file_name)

        data = {
        "fullname":fullname,
        "email": email,
        "gender":gender,
        "filename": new_file_name}

    return render_template('add_meeting_form.html')

def save_speaker(speaker):
    with open(SPEAKER_DATA_PATH) as data_file:   
        print "save oo!" 
        speakers = json.load(data_file)
    print "save tooo!" 
    next_index = len(speakers.keys())
    speakers[next_index] = speaker
    print "save waaa!" 
    with open(SPEAKER_DATA_PATH, 'w') as outfile:
        json.dump(speakers, outfile)

@app.route("/process_file/")
def process_file_into_meeting():
    pass
        

if __name__ == "__main__":
    app.run()

