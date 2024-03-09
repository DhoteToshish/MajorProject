import pandas as pd
import plotly.express as px
import numpy as np
def load_data(year, type):
    data = pd.read_excel(f"xlsxFiles/{type}/{year}.xlsx")
    dataFrame = getDataFrame(data)
    return dataFrame

def getDataFrame(excleData):
    numeric_mean = excleData.mean(numeric_only=True)
    excleData.fillna(numeric_mean, inplace=True)
    pd.set_option('display.max_rows', None)
    dataFrame = excleData
    return dataFrame

def getcomparisonAcrossLocationData():
    data = pd.read_excel(f"xlsxFiles/air_pollution_dataset.xlsx")
        
    data.replace('-', np.nan, inplace=True)

    data = data.replace([np.inf, -np.inf], np.nan)

    float_columns = data.select_dtypes(include=[np.float64]).columns
    data[float_columns] = data[float_columns].fillna(0).astype(int)
    return data


# def comparisonAcrossLocationsSO2():
#     data = getcomparisonAcrossLocationData()

#     so2_data = data[['State/UT', 'Annual Average SO2']].copy()
#     location_so2_mean = so2_data.groupby('State/UT')['Annual Average SO2'].mean().sort_values().reset_index()

#     fig = px.bar(location_so2_mean, x='Annual Average SO2', y='State/UT', orientation='h',
#                 title='Mean SO2 Levels Across Different States/UT',
#                 labels={'Annual Average SO2': 'Mean SO2 Levels', 'State/UT': 'State/UT'})
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)


# def comparisonAcrossLocationsNO2():
#     data = getcomparisonAcrossLocationData()
#     no2_data = data[['State/UT', 'Annual Average NO2']].copy()

    
#     location_no2_mean = no2_data.groupby('State/UT')['Annual Average NO2'].mean().sort_values().reset_index()

    
#     fig = px.bar(location_no2_mean, x='Annual Average NO2', y='State/UT', orientation='h',
#                 title='Mean NO2 Levels Across Different States/UT',
#                 labels={'Annual Average NO2': 'Mean NO2 Levels', 'State/UT': 'State/UT'})
#     return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)

# def comparisonAcrossLocationsPM10():
    data = getcomparisonAcrossLocationData()
    pm10_data = data[['State/UT', 'Annual Average PM10']].copy()

    
    location_pm10_mean = pm10_data.groupby('State/UT')['Annual Average PM10'].mean().sort_values().reset_index()

   
    fig = px.bar(location_pm10_mean, x='Annual Average PM10', y='State/UT', orientation='h',
                title='Mean PM10 Levels Across Different States/UT',
                labels={'Annual Average PM10': 'Mean PM10 Levels', 'State/UT': 'State/UT'})
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)

def comparisonAcrossLocations(pollutant):
    data = getcomparisonAcrossLocationData()
    pollutant_data = data[['State/UT', f'Annual Average {pollutant}']].copy()
    location_pollutant_mean = pollutant_data.groupby('State/UT')[f'Annual Average {pollutant}'].mean().sort_values().reset_index()

    fig = px.bar(location_pollutant_mean, x=f'Annual Average {pollutant}', y='State/UT', orientation='h',
                title=f'Mean {pollutant} Levels Across Different States/UT',
                labels={f'Annual Average {pollutant}': f'Mean {pollutant} Levels', 'State/UT': 'State/UT'})
    
    SO2 = {
        "sentences" : ["The x-axis represents the mean SO2 levels, ranging from 0 to 20.",
                       "The y-axis lists the names of different states and union territories.",
                       "Each state or UT is represented by a blue bar extending to the right, indicating its respective mean annual average SO2 level.",
                       "Jharkhand has the highest recorded mean annual average SO2 level, followed by Daman & Diu, Dadra & Nagar Haveli, and others.",
                       "Lakshadweep has the lowest mean annual average SO2 level on this graph."
                       ]
    }

    PM10 ={
        "sentences":[
            "The x-axis represents the mean PM10 levels, ranging from 0 to 200.",
            "The y-axis lists the names of different states and union territories.",
            "Each state or UT is represented by a blue bar extending to the right, indicating its respective mean annual average PM10 level.",
            "Delhi has the longest bar, indicating it has the highest mean PM10 level among the listed regions.",
            "Manipur has one of the shortest bars, showing it has one of the lowest levels of mean annual average PM10."
        ]
    }

    NO2 = {
        "sentences":[
            "The x-axis represents the mean NO2 levels, ranging from 0 to 60.",
            "The y-axis lists the names of different states and union territories.",
            "Each state or UT is represented by a blue bar extending to the right, indicating its respective mean annual average NO2 level.",
            "Delhi has the highest recorded mean annual average NO2 level, followed by Jharkhand and Haryana.",
            "Lakshadweep has the lowest m5.	Lakshadweep has one of the lowest mean annual average NO2 levels on this graph."
        ]
    }

    sentences = []
    if pollutant == "SO2":
        sentences = SO2["sentences"]
    elif pollutant == "NO2":
        sentences = NO2["sentences"]
    else:
        sentences = PM10['sentences']
    
    
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), pollutant, sentences ]