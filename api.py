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
    app.run(debug=True)
