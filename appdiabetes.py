from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load("trainedmodel.joblib")

# Create a Flask app
app = Flask(__name__)

# Define the homepage route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get user input from the form
    highbp = int(request.form["highbp"])
    highchol = int(request.form["highchol"])
    cholcheck = int(request.form["cholcheck"])
    bmi = float(request.form["bmi"])
    smoker = int(request.form["smoker"])
    stroke = int(request.form["stroke"])
    heartdisease = int(request.form["heartdisease"])
    physactivity = int(request.form["physactivity"])
    veggies = int(request.form["veggies"])
    hvalcohol = int(request.form["hvalcohol"])
    genhlth = int(request.form["genhlth"])
    menthlth = int(request.form["menthlth"])
    physhlth = int(request.form["physhlth"])
    diffwalk = int(request.form["diffwalk"])
    age_discreet = int(request.form["age"])
    age =int()
    if 18 <= age_discreet <= 24:
        age = 1
    elif 25 <= age_discreet <= 29:
        age = 2
    elif 30 <= age_discreet <= 34:
        age = 3
    elif 35 <= age_discreet <= 39:
        age = 4
    elif 40 <= age_discreet <= 44:
        age = 5
    elif 45 <= age_discreet <= 49:
        age = 6
    elif 50 <= age_discreet <= 54:
        age = 7
    elif 55 <= age_discreet <= 59:
        age = 8
    elif 60 <= age_discreet <= 64:
        age = 9
    elif 65 <= age_discreet <= 69:
        age = 10
    elif 70 <= age_discreet <= 74:
        age = 11
    elif 75 <= age_discreet <= 79:
        age = 12
    else:
        age = 13
    education = int(request.form["education"])
    income_discreet = int(request.form["income"])
    if income_discreet < 10000:
        income = 1
    elif income_discreet < 15000:
        income = 2
    elif income_discreet < 25000:
        income = 3
    elif income_discreet < 35000:
        income = 4
    elif income_discreet < 50000:
        income = 5
    elif income_discreet < 75000:
        income = 6
    else:
        income = 8 
    # Prepare the user input as a feature vector
    user_input = [highbp, highchol, cholcheck, bmi, smoker, stroke, heartdisease,
                  physactivity, veggies, hvalcohol, genhlth, menthlth, physhlth, diffwalk, age,
                  education, income]

    # Make prediction using the model
    prediction = model.predict(np.array([user_input]))

    # Interpret the prediction (0: no diabetes, 1: diabetes)
    if prediction[0] == 0:
        result = "Negative. The model predicts you might not have diabetes."
    else:
        result = "Positive. The model predicts you might have diabetes."

    # Render the prediction page with the result
    return render_template("prediction.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
