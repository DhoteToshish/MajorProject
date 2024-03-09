from flask import Flask, render_template
from data_processing import comparisonAcrossLocations,load_data
import pandas as pd
app = Flask(__name__, static_folder='static')

@app.route("/")
def home():
    return render_template("home.html")

# @app.route("/air_data_analysis")
# def air_data_analysis():
#     return render_template("airDataAnalysis.html")

# @app.route("/water_data_analysis")
# def water_data_analysis():
#     return render_template("waterDataAnalysis.html")

# @app.route("/noise_data_analysis")
# def noise_data_analysis():
#     return render_template("noiseDataAnalysis.html")

@app.route("/data_analysis/<type>")
def data_analysis(type):
    if type == 'air':
        return render_template("airDataAnalysis.html")
    elif type == 'water':
        return render_template("waterDataAnalysis.html")
    elif type== 'noise':
        return render_template("noiseDataAnalysis.html")

@app.route("/air_dataset_yearlist")
def air_dataset_yearlist():
    data = None
    return render_template("airDataSetYearList.html", data=data)

@app.route("/water_dataset_yearlist")
def water_dataset_yearlist():
    return render_template("waterDataSetYearList.html")

@app.route("/noise_dataset_yearlist")
def noise_dataset_yearlist():
    data = None
    return render_template("noiseDataSetYearList.html", data=data)

@app.route("/AboutUs")
def AboutUs():
    return render_template("AboutUs.html")

@app.route("/working")
def working():
    return render_template("working.html")

@app.route("/air_dataset_yearlist/<year>/<type>")
def air_dataset_year(year,type):
    data = load_data(year=year, type=type)
    return render_template("airDataSetYearList.html", data=data)

@app.route("/noise_dataset_yearlist/<year>/<type>")
def noise_dataset_year(year,type):
    data = load_data(year=year, type=type)
    return render_template("noiseDataSetYearList.html", data=data)

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/comparisonAcrossLocations/<pollutant>")
def comparisonsAcrossLocations(pollutant):
    data = comparisonAcrossLocations(pollutant=pollutant)
    return render_template("CAL_Pollutant.html", data = data[0], pollutant = data[1], sentences = data[2] )


if __name__ == "__main__":
    app.run(debug=True)