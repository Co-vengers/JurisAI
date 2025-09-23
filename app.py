import os
import json
from utils import Utils
from flask import Flask

app = Flask(__name__)
app.secret_key = os.urandom(24)

util = Utils()

if __name__ == '__main__':
    util.preprocess_and_embed_data()
    app.run(debug=True)