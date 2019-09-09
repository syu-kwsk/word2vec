import os
import io
import time
import numpy as np
# import cv2
# import dlib
from flask import render_template, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from gensim.models import Word2Vec
from flask import Flask, jsonify, request
import nltk

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/')
def word2vec():
    model = Word2Vec.load("./wiki.model")
    w = request.args['word']
    return jsonify(model.most_similar(positive=[w], topn=10))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
