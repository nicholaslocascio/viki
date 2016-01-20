from flask import Flask
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug import secure_filename
import json
import uuid
import play



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
        "title":title,
        "filename": new_file_name}

        data_id = save_meeting(data)
        payload = data_id
        return redirect(url_for('show_meeting', payload=payload) + data_id)
    return render_template('add_meeting_form.html')

@app.route('/show_meeting/<meeting_id>', methods=['GET'])
def show_meeting(meeting_id):
    print 'meeet', meeting_id
    meeting = get_meeting(meeting_id)
    return render_template('show_meeting.html', meeting=meeting)

MEETING_DATA_PATH = 'data/meetings.json'


def get_percentages(tups):
    pass

def save_meeting(meeting):
    filename = meeting["filename"]
    predictions = play.get_all_new_predictions_for_meeting(filename)
    meeting["predictions"] = predictions
    meeting["percentages"] = get_tuples(predictions)

    save_meeting_to_db(meeting)

    return data_id

def save_meeting_to_db(meeting):
    with open(MEETING_DATA_PATH) as data_file:   
        print "save oo!" 
        meetings = json.load(data_file)
    print "save tooo!" 
    next_index = len(speakers.keys())
    meetings[next_index] = meeting
    print "save waaa!" 
    with open(MEETING_DATA_PATH, 'w') as outfile:
        json.dump(meetings, outfile)
    return next_index

def get_meeting(data_id):

    default_meeting = {
    "title": "Important meeting for Investors",
    "percentages": [
    ["18.6", {"username":"Nick Locascio",
    "gender": "Male",
    }],
    ["33.4", 
    {"username":"Harini Suresh",
    "gender": "Female"
    }],
    ["25.4", 
    {"username":"Nadia Wallace",
    "gender": "Female"
    }],
    ["10.4", 
    {"username":"Kate Candon",
    "gender": "Female"
    }],
    ],
    "interruptions": [25, 75, 100, 0], # MTM, MTF, FTF, FTM
    "interruptions_count": 12
    }

    if False:
        return default_meeting

    data_id = 0

    meetings = get_meetings()
    print "MEEEEET", meetings
    meeting = meetings.get(data_id, default_meeting)
    return meeting

def get_meetings():
    meetings = []
    with open('data/meetings.json') as data_file:   
        print "save oo!" 
        print data_file
        meetings = json.load(data_file)
    return meetings

def process_predictions_into_percentages(predictions):
    percentages = {}
    for p in predictions:
        percentages[p] = percentages.get(p, 0) + 1
    return percentages

def get_tuples(predictions):
    tups = {}
    speakers = get_speakers()
    for key, value in percentages.iteritems():
        speaker = speakers[key]
        tups[key] = [speaker, value]
    return tups

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

def get_speakers():
    with open(SPEAKER_DATA_PATH) as data_file:   
        print "save oo!" 
        speakers = json.load(data_file)
    return speakers

if __name__ == "__main__":
    app.run()

