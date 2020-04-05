from flask import render_template, request
from app import app
from app.forms import TextSumForm
import requests
import json
from config import API_ENDPOINT

url = API_ENDPOINT.URL
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    summarised_text = ['Appear here']
    form = TextSumForm()
    if request.method == 'POST':
        full_text = request.form['inputText']
        data = {"text":
           #"This aosad asdhs das asd a. rthrwf rthwf aefg rocoker ys. adthe ys ys ys habh. bothwe thase athwear ehr this is the tird sentence and this is the last sentence and this is the final sentence and this is the one sentence and this is the waiting sent and this"
            full_text
        } 
        r = requests.post(url = url, data = json.dumps(data))
        summarised_text = json.loads(r.text)['summary']
        summarised_text = summarised_text.split('\n')
        print(summarised_text)
        return render_template('index.html', title='Home', form=form, summary = summarised_text)
    else:
        return render_template('index.html', title='Home', form=form, summary = summarised_text)

