from flask import Flask, render_template
from data_processing import load_data
import pandas as pd
app = Flask(__name__, static_folder='static')

@app.route("/")
def home():
    data = load_data()
    return render_template("home.html", data=data)

@app.route("/air_data_analysis")
def air_data_analysis():
    return render_template("airDataAnalysis.html")

@app.route("/water_data_analysis")
def water_data_analysis():
    return render_template("waterDataAnalysis.html")

@app.route("/noise_data_analysis")
def noise_data_analysis():
    return render_template("noiseDataAnalysis.html")

@app.route("/air_dataset_yearlist")
def air_dataset_yearlist():
    data = load_data()
    print("data", data)
    return render_template("airDataSetYearList.html", data=data)

@app.route("/water_dataset_yearlist")
def water_dataset_yearlist():
    return render_template("waterDataSetYearList.html")

@app.route("/noise_dataset_yearlist")
def noise_dataset_yearlist():
    return render_template("noiseDataSetYearList.html")

@app.route("/AboutUs")
def AboutUs():
    return render_template("AboutUs.html")

@app.route("/working")
def working():
    return render_template("working.html")
    
    
if __name__ == "__main__":
    app.run(debug=True)