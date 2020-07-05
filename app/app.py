import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=["POST"])
def predict_fun():
    
    NB_hoax_model = open('NB_hoax_model.pkl', 'rb')
    clf = joblib.load(NB_hoax_model)
    
    cv_model = open('cv.pkl', 'rb')
    cv = joblib.load(cv_model)

    if request.method == "POST":

        message = request.form['berita']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
