import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=["POST"])
def predict():
    
    NB_hoax_model = open('app/NB_hoax_model.pkl', 'rb')
    clf = joblib.load(NB_hoax_model)
    
    cv_model = open('app/cv.pkl', 'rb')
    cv = joblib.load(cv_model)

    if request.method == "POST":

        message = request.form['berita']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if my_prediction == 1:
            result = "Hoax!"
            
        else:
            result = "Valid."

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
