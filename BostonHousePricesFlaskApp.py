# import the necessary dependencies
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# create the Flask application object and assign it to the variable 'app'
app = Flask(__name__)


# load the 'BostonHousePrices_LinearRegressionModel_StandardScaler.pkl' file
import_directory = r"/Users/ncheymbamalu/Desktop/Data_Science/ML_EndToEnd_Projects/Boston_House_Prices/"
pickle_file = r"BostonHousePrices_LinearRegressionModel_StandardScaler.pkl"
pickle_objects = pickle.load(open(import_directory + pickle_file, "rb"))
linreg_model = pickle_objects["linear_regression_model"]
scaler = pickle_objects["standard_scaler"]


# Flask application's home page
# important note 1, a "templates" directory and "home.html" file must be created within the present working directory
# important note 2, the 'home.html' file is where the Flask application's homepage will be created/edited
@app.route("/")
def home():
    return render_template("home.html")


# Flask application's prediction API (application programming interface)
# the Postman API platform will be used to generate predictions
# for step-by-step instructions, refer to this YouTube link: https://youtu.be/MJ1vWb1rGwM
# to determine if a json object (dictionary) is valid: https://jsonlint.com/
@app.route("/predict_api", methods=["POST"])
def predict_api():
    # read in a single record as a json object; assign it to the variable 'record'
    # important note, the json object is a dictionary whose keys are features and...
    # ...whose values are the actual data (see the Postman API platform for an example of a single record)
    record = request.json["data"]

    # extract the dictionary keys, i.e., the features, from the 'record' dictionary
    features = list(record.keys())

    # 1) extract the dictionary values from the 'record' dictionary,
    # 2) store the values in a list
    # 3) convert the list to a NumPy array
    # 4) reshape the NumPy array to (1, D), where D is the number of features
    record = np.array(list(record.values())).reshape(1, -1)

    # print out the 'record' NumPy array
    print(record)

    # convert the 'record' NumPy array to a Pandas DataFrame
    record = pd.DataFrame(record, columns=features)

    # standardize the 'record' DataFrame by calling the 'scaler' object
    record_scaled = scaler.transform(record)

    # make a prediction on the 'record_scaled' NumPy array by calling the 'linreg_model' object
    prediction = np.round(linreg_model.predict(record_scaled), 4)

    # print out the prediction
    print(prediction[0])

    # convert the 'prediction' NumPy array to a json object
    return jsonify(prediction[0])


# Flask front-end web application, i.e., the web page where predictions are made
# note, this where values can be input for each feature (see the "home.html" file inside the 'templates' directory)
@app.route("/predict", methods=["POST"])
def predict():
    # read in a single record as a list of floats
    record = [float(x) for x in request.form.values()]

    # specify the features
    features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'b', 'lstat']

    # 1) convert the 'record' list to a NumPy array
    # 2) reshape the NumPy array to (1, D), where D is the number of features
    # 3) convert the (1, D) NumPy array to a DataFrame and set the 'columns' parameter equal to the 'features' list
    record = pd.DataFrame(np.array(record).reshape(1, -1), columns=features)

    # standardize the 'record' DataFrame by calling the 'scaler' object
    record_scaled = scaler.transform(record)

    # make a prediction on the 'record_scaled' NumPy array by calling the 'linreg_model' object
    prediction = linreg_model.predict(record_scaled)[0]

    # output the prediction in $1000s
    # note, refer to the "home.html" file for the 'render_template' object's 'prediction_text' parameter
    return render_template("home.html", prediction_text=f"The predicted median home price is ${(prediction * 1000):.2f}")


if __name__ == "__main__":
    app.run(debug=True)