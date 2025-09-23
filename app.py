import os
import json
from utils import Utils
import google.generativeai as genai
from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.urandom(24)

util = Utils()

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        user_query = request.form.get('user_query')
        if user_query:
            bot_response = util.get_chatbot_response(user_query)
            session['history'].append({'question': user_query, 'answer': bot_response})
            session.modified = True
        return redirect(url_for('index'))

    return render_template('index.html', conversation_history=session['history'])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    util.preprocess_and_embed_data()
    app.run(debug=True)