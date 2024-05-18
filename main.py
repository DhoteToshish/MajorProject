from flask import Flask, render_template
from data_processing import comparisonAcrossLocations,distributionsOfPollutionLevels,OutLierDetectionOfAir,InterActiveDashBoard,ComplianceAssessmentOfWaterQuality,StateWiseDisolvedOxygenRange,StateWiseWaterPh,StateWiseWaterTemperature,getPredictivemodel,ComparativeAnalysisOfWaterModel,DiurnalLimitsTrendOfNoise,disparityAcrossAreaForNoise,GetPredictiveModelOfNoise,ComparativeAnalysisOfNoiseModel,getWaterDataSet,ComparativeAnalysisOfAirModel,GetPredictiveModelOfAir,load_data
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

@app.route("/distributionOfPollutionLevels")
def distributionOfPollutionLevels():
    data = distributionsOfPollutionLevels()
    return render_template("distributionOfPollutionLevel.html",data = data)

@app.route("/outLierDetectionOfAir")
def outLierDetectionOfAir():
    data = OutLierDetectionOfAir()
    return render_template("outLierDetectionOfAir.html",data=data)

@app.route("/interActiveDashBoard")
def interActiveDashBoard():
    data = InterActiveDashBoard()
    return render_template("InterActiveDashBoard.html", data=data)


@app.route("/insightsOfWater")
def insightsOfWater():
    return render_template("InsightsOfWater.html")

@app.route("/ComplianceAssessmentofwaterquality")
def ComplianceAssessmentofwaterquality():
    data = ComplianceAssessmentOfWaterQuality()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/StateWiseDORange")
def StateWiseDORange():
    data = StateWiseDisolvedOxygenRange()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/StateWisewaterPh")
def StateWisewaterPh():
    data = StateWiseWaterPh()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/StateWisewatertemp")
def StateWisewatertemp():
    data = StateWiseWaterTemperature()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/PredictionofWaterQualityIndex")
def PredictionofWaterQualityIndex():
    return render_template("PredictionofWaterQualityIndex.html")

@app.route("/getPredictiveModel/<model>")
def getPredictiveModel(model):
    data = getPredictivemodel(model)
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = data[3],modelTorender = data[4], metric_df = None, preDictiveDf = None)


@app.route("/ComparativeAnalysisOfWater")
def ComparativeAnalysisOfWater():
    data = ComparativeAnalysisOfWaterModel()
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = None,modelTorender = None, metric_df = data[5], preDictiveDf = None)
    
@app.route("/insightsOfNoise")
def insightsOfNoise():
    return render_template("InsightsOfNoise.html")

@app.route("/DiurnalLimitsTrend")
def DiurnalLimitsTrend():
    data = DiurnalLimitsTrendOfNoise()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/DisparityAcrossAreaForNoise")
def DisparityAcrossAreaForNoise():
    data = disparityAcrossAreaForNoise()
    return render_template("ComplianceAssessmentOfWaterQuality.html", data = data[0], heading = data[1], sentences = data[2])

@app.route("/PredictionofNoiseQuality")
def PredictionofNoiseQuality():
    return render_template("PredictionofNoiseQuality.html")

@app.route("/getPredictiveModelOfNoise/<model>")
def getPredictiveModelOfNoise(model):
    data = GetPredictiveModelOfNoise(model)
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = None,modelTorender = data[4], metric_df = None, preDictiveDf = data[6])


@app.route("/ComparativeAnalysisOfNoise")
def ComparativeAnalysisOfNoise():
    data = ComparativeAnalysisOfNoiseModel()
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = None,modelTorender = None, metric_df = data[5], preDictiveDf = None)


@app.route("/waterDataSet")
def  waterDataSet():
    data = getWaterDataSet()
    return render_template("waterDataSet.html",data=data)

@app.route("/predictionOfAirQuality")
def predictionOfAirQuality():
    return render_template("PredictionofAirQuality.html")

@app.route("/getPredictiveModelOfAir/<model>")
def getPredictiveModelOfAir(model):
    data = GetPredictiveModelOfAir(model)
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = None,modelTorender = data[4], metric_df = None, preDictiveDf = data[6])

@app.route("/ComparativeAnalysisOfAir")
def ComparativeAnalysisOfAir():
    data = ComparativeAnalysisOfAirModel()
    return render_template("displayPredictiveModel.html",data = data[0],heading = data[1], sentences = data[2], df = None,modelTorender = None, metric_df = data[5], preDictiveDf = None)

if __name__ == "__main__":
    app.run(debug=True)