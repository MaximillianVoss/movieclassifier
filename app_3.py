from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import numpy as np, os, pickle, sqlite3
import dill
from vectorizer import vect
app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
# clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
clf = dill.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
def classify(document):
    label = ['negative', 'positive']
    x = vect.transform([document])
    y = clf.predict(x)[0]
    proba = np.max(clf.predict_proba(x))
    return label[y], proba
def train(document, y):
    x = vect.transform([document])
    clf.partial_fit(x, [y])
def sqlite_entry(path, document, y):
    conn = sqlite.connect(path)
    c = conn.cursor()
    c.execute("insert into review_db (review, sentiment, date) values (?, ?, datetime('now'))", (document, y))
    conn.commit()
    conn.close()
class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])
# C:\\Users\\FossW\\OneDrive\\Рабочий стол\\movieclassifier\\reviewform.html
@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)
@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html', content=review, prediction=y, probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)
@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    y = label.index(prediction)
    if feedback == 'Incorrect':
        y = int(not y)
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')
if __name__ == '__main__':
    app.run(debug=True)
