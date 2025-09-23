from app import app
from flask import session, request, redirect, url_for, render_template
from app import util

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