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
        # Replace '-' with NaN
    data.replace('-', np.nan, inplace=True)

    # Convert non-finite values to NaN
    data = data.replace([np.inf, -np.inf], np.nan)

# Convert float columns to integers
    float_columns = data.select_dtypes(include=[np.float64]).columns
    data[float_columns] = data[float_columns].fillna(0).astype(int)
    return data
def comparisonAcrossLocationsSO2():
    data = getcomparisonAcrossLocationData()

    so2_data = data[['State/UT', 'Annual Average SO2']].copy()
    location_so2_mean = so2_data.groupby('State/UT')['Annual Average SO2'].mean().sort_values().reset_index()

    fig = px.bar(location_so2_mean, x='Annual Average SO2', y='State/UT', orientation='h',
                title='Mean SO2 Levels Across Different States/UT',
                labels={'Annual Average SO2': 'Mean SO2 Levels', 'State/UT': 'State/UT'})
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)


def comparisonAcrossLocationsNO2():
    data = getcomparisonAcrossLocationData()
    no2_data = data[['State/UT', 'Annual Average NO2']].copy()

    # Group data by location and calculate mean NO2 levels
    location_no2_mean = no2_data.groupby('State/UT')['Annual Average NO2'].mean().sort_values().reset_index()

    # Plotting
    fig = px.bar(location_no2_mean, x='Annual Average NO2', y='State/UT', orientation='h',
                title='Mean NO2 Levels Across Different States/UT',
                labels={'Annual Average NO2': 'Mean NO2 Levels', 'State/UT': 'State/UT'})
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)

def comparisonAcrossLocationsPM10():
    data = getcomparisonAcrossLocationData()
    pm10_data = data[['State/UT', 'Annual Average PM10']].copy()

    # Group data by location and calculate mean PM10 levels
    location_pm10_mean = pm10_data.groupby('State/UT')['Annual Average PM10'].mean().sort_values().reset_index()

    # Plotting
    fig = px.bar(location_pm10_mean, x='Annual Average PM10', y='State/UT', orientation='h',
                title='Mean PM10 Levels Across Different States/UT',
                labels={'Annual Average PM10': 'Mean PM10 Levels', 'State/UT': 'State/UT'})
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)