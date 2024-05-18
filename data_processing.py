import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots       
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
app = dash.Dash(__name__)


# Define permissible limits and weights for each parameter
parameter_limits = {
    'Temperature (Celsius) min': 28,
    'Temperature (Celsius) max': 28,
    'Dissolved O2 min': 7.5,
    'Dissolved O2 max': 7.5,
    'pH min': 8.5,
    'pH max': 8.5,
    'Conductivity (µmhos/cm) min': 150,
    'Conductivity (µmhos/cm) max': 150,
    'B.O.D. min': 3,
    'B.O.D. max': 3,
    'Nitrate-N + Nitrite-N (mg/l) min': 0.503,
    'Nitrate-N + Nitrite-N (mg/l) max': 0.503,
    'Total Coli form (MPN/100ml) min': 100,
    'Total Coli form (MPN/100ml) max': 100,
    'Faecal Coli form (MPN/100ml) min': 60,
    'Faecal Coli form (MPN/100ml) max': 60
}

parameter_weights = {
    'Temperature (Celsius) min': 0.035714286,
    'Temperature (Celsius) max': 0.035714286,
    'Dissolved O2 min': 0.133333333,
    'Dissolved O2 max': 0.133333333,
    'pH min': 0.117647059,
    'pH max': 0.117647059,
    'Conductivity (µmhos/cm) min': 0.006666667,
    'Conductivity (µmhos/cm) max': 0.006666667,
    'B.O.D. min': 0.333333333,
    'B.O.D. max': 0.333333333,
    'Nitrate-N + Nitrite-N (mg/l) min': 1.988071571,
    'Nitrate-N + Nitrite-N (mg/l) max': 1.988071571,
    'Total Coli form (MPN/100ml) min': 0.01,
    'Total Coli form (MPN/100ml) max': 0.01,
    'Faecal Coli form (MPN/100ml) min': 0.016666667,
    'Faecal Coli form (MPN/100ml) max': 0.016666667
}

# Define water quality descriptions based on WQI ranges
def get_water_quality_description(wqi_value):
    if wqi_value < 50:
        return 'Excellent, Non-polluted'
    elif 50 <= wqi_value < 100:
        return 'Good Water, Non-polluted'
    elif 100 <= wqi_value < 200:
        return 'Poor Water, Polluted'
    elif 200 <= wqi_value < 300:
        return 'Very Poor Water, Polluted'
    else:
        return 'Water Unsuitable for Drinking, Heavily Polluted'

# Define Water Quality Index (WQI) Calculation
def calculate_wqi(row):
    wqi_values = []
    
    for param in parameter_limits:
        value = row[param]
        limit = parameter_limits[param]
        weight = parameter_weights[param]
        
        if value <= limit:
            wqi_value = 100
        else:
            wqi_value = 100 * (1 - (value - limit) / value)
        
        wqi_values.append(wqi_value * weight)
    
    wqi = sum(wqi_values)
    
    return wqi


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


def distributionsOfPollutionLevels():
    data = pd.read_excel(f"xlsxFiles/air_pollution_dataset.xlsx")
        
    data.replace('-', np.nan, inplace=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    # List of pollutants
    pollutants = ['Annual Average SO2', 'Annual Average NO2', 'Annual Average PM10']

    # Define colors for the bars
    colors = px.colors.qualitative.Set1

    # Create histograms for each pollutant with different colors for each bar
    histograms = []
    for i, pollutant in enumerate(pollutants):
        hist = px.histogram(data, x=pollutant, nbins=30, color_discrete_sequence=[colors[i]],
                            marginal='box', title='Distribution of ' + pollutant)
        histograms.append(hist.to_html(full_html=False, include_plotlyjs='cdn'))

    return histograms
      

def OutLierDetectionOfAir():
    data = pd.read_excel(f"xlsxFiles/air_pollution_dataset.xlsx")
        
    data.replace('-', np.nan, inplace=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    # Selecting the relevant columns for outlier detection
    pollutants = ['Annual Average SO2', 'Annual Average NO2', 'Annual Average PM10']
    pollution_data = data[pollutants]

    # Box plot for outlier detection
    boxplot_fig = px.box(pollution_data, title="Box Plot of Air Pollutants", points="all")
    boxplot_fig.update_traces(marker=dict(color='rgba(0,0,0,0.5)', size=2),
                            line=dict(color='rgba(0,0,0,0.5)'))
    boxplot_fig.update_layout(xaxis_title="Pollutants", yaxis_title="Pollution Level")

    return  [boxplot_fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500)]


def generate_figure(selected_pollutant):
    data = pd.read_excel(f"xlsxFiles/air_pollution_dataset.xlsx")
        
    data.replace('-', np.nan, inplace=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    fig = px.line(data, x=data.index, y=selected_pollutant, title=f'{selected_pollutant} Levels Over Time')
    return fig

# Define the layout of the dashboard
app.layout = html.Div([
    dcc.Dropdown(
        id='pollutant-dropdown',
        options=[
            {'label': 'SO2', 'value': 'Annual Average SO2'},
            {'label': 'NO2', 'value': 'Annual Average NO2'},
            {'label': 'PM10', 'value': 'Annual Average PM10'},
        ],
        value='Annual Average PM10',
        clearable=False
    ),
    dcc.Graph(id='pollution-graph')
])

# Define callback to update the graph based on dropdown selection
@app.callback(
    Output('pollution-graph', 'figure'),
    [Input('pollutant-dropdown', 'value')]
)
def update_graph(selected_pollutant):
    return generate_figure(selected_pollutant)

def InterActiveDashBoard():
    generate_figure('Annual Average ')


def ComplianceAssessmentOfWaterQuality():
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))

    # Define threshold values for compliance (example thresholds, replace with actual standards)
    thresholds = {
        'Dissolved O2 min': 4.0,  # Minimum Dissolved Oxygen threshold (mg/l)
        'pH min': 6.5,  # Minimum pH threshold
        'B.O.D. min': 2.0  # Maximum BOD threshold (mg/l)
    }

    # Check compliance status for each parameter
    df['Dissolved O2 Compliance'] = df['Dissolved O2 min'] >= thresholds['Dissolved O2 min']
    df['pH Compliance'] = df['pH min'] >= thresholds['pH min']
    df['BOD Compliance'] = df['B.O.D. min'] <= thresholds['B.O.D. min']

    # Determine overall compliance status for each station
    df['Compliance'] = df[['Dissolved O2 Compliance', 'pH Compliance', 'BOD Compliance']].all(axis=1)

    # Create compliance assessment scatter plot using Plotly
    fig = px.scatter(df, x='Station Name', color='Compliance', hover_name='Station Name',
                    color_discrete_map={True: '#86af49', False: '#e0876a'},
                    title='Compliance Assessment of Water Quality at Various Stations',
                    labels={'Station Name': 'Station Name', 'Compliance': 'Compliance Status'})

    # Customize layout for better visualization
    #fig.update_xaxes(,tickfont=dict(size=10), title=None)  # Rotate x-axis labels for readability
    fig.update_yaxes(title='Compliance Status')
    sentences = [
        "Models: The x-axis represents different station names where water quality is assessed.",
        "The y-axis ranges from 0 to 1000.",
        "Each station is represented by a bar.",
        "Red bars indicate non-compliance (False) with water quality standards.",
	    "Green bars indicate compliance (True) with water quality standards.",
        "In summary, most stations are not in compliance with water quality standards, as indicated by the predominance of red bars"
    ]
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500),'Compliance Assessment of Water Quality at Various Stations', sentences]

def StateWiseDisolvedOxygenRange():
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))

    # Calculate average min and max Dissolved O2 by state
    avg_do_by_state = df.groupby('State Name').agg({
        'Dissolved O2 min': 'mean',
        'Dissolved O2 max': 'mean'
    }).reset_index()

    # Create grouped bar chart with custom colors using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=avg_do_by_state['State Name'],
        y=avg_do_by_state['Dissolved O2 min'],
        name='Average Min Dissolved O2',
        marker_color='#b0aac0'  # Custom color for min Dissolved O2
    ))

    fig.add_trace(go.Bar(
        x=avg_do_by_state['State Name'],
        y=avg_do_by_state['Dissolved O2 max'],
        name='Average Max Dissolved O2',
        marker_color='#77a8a8'  # Custom color for max Dissolved O2
    ))

    # Update layout for better visualization
    fig.update_layout(
        barmode='group',
        title='State-wise Average Minimum and Maximum Dissolved Oxygen (D.O.)',
        xaxis_title='State',
        yaxis_title='Dissolved Oxygen (mg/l)',
        legend_title='Dissolved O2 Type',
        width=1450,  # Set width of the plot (adjust as needed)
        height=800,  # Set height of the plot (adjust as needed)
        margin=dict(l=60, r=60, t=90, b=110)  # Adjust margin to provide space for labels
    )

    # Find states with the lowest and highest average Dissolved O2 values
    state_with_lowest_min_do = avg_do_by_state.loc[avg_do_by_state['Dissolved O2 min'].idxmin()]['State Name']
    state_with_highest_max_do = avg_do_by_state.loc[avg_do_by_state['Dissolved O2 max'].idxmax()]['State Name']

    # Add annotations for lowest and highest average Dissolved O2 values
    fig.add_annotation(
        x=state_with_lowest_min_do, y=avg_do_by_state['Dissolved O2 min'].min(),
        text=f"Lowest Min Dissolved O2 ({avg_do_by_state['Dissolved O2 min'].min():.2f} mg/l)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    fig.add_annotation(
        x=state_with_highest_max_do, y=avg_do_by_state['Dissolved O2 max'].max(),
        text=f"Highest Max Dissolved O2 ({avg_do_by_state['Dissolved O2 max'].max():.2f} mg/l)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )
    sentences = [
        "The x-axis is labeled with the names of various states.",
	"The y-axis represents Dissolved Oxygen (D.O.) levels measured in mg/l (milligrams per liter), ranging from 0 to 9 mg/l.",
	"Each state has two bars side by side: one indicates the average minimum dissolved oxygen level (dark green), and the other indicates the average maximum dissolved oxygen level (light green).",
	"Annotations are present indicating the “Highest Max Dissolved O2” at approximately 7.90 mg/l and the “Lowest Min Dissolved O2” close to 0.20 mg/l.",
    "The graph helps us understand the variation of dissolved oxygen levels across different states in India.",
	"Some states have higher dissolved oxygen levels (closer to 7.90 mg/l), while others have lower levels (near 0.20 mg/l)"
    ]
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), 'State-wise Average Minimum and Maximum Dissolved Oxygen (D.O.)', sentences]

def StateWiseWaterPh():
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))

    # Calculate average min and max pH by state
    avg_ph_by_state = df.groupby('State Name').agg({
        'pH min': 'mean',
        'pH max': 'mean'
    }).reset_index()

    # Plotting using Plotly with custom colors for min and max bars
    fig = px.bar(avg_ph_by_state, x='State Name', y=['pH min', 'pH max'],
                barmode='group', labels={'value': 'pH'}, 
                title='State-wise Average Minimum and Maximum Water pH')

    # Define custom colors for the bars (min: blue, max: orange)
    color_map = {'pH max': '#6b5b95', 'pH min': '#a2b9bc'}

    # Update layout and color mapping for better visualization
    fig.update_layout(
        xaxis_title='State',
        yaxis_title='pH',
        legend_title='pH Type',
        width=1400,  # Set width of the plot (adjust as needed)
        height=700,  # Set height of the plot (adjust as needed)
        margin=dict(l=60, r=60, t=90, b=110)  # Adjust margin to provide space for labels
    )

    # Apply custom colors to the bars
    for trace, color in color_map.items():
        fig.update_traces(marker_color=color, selector=dict(name=trace))

    # Find states with the lowest and highest average pH values
    state_with_lowest_min_ph = avg_ph_by_state.loc[avg_ph_by_state['pH min'].idxmin()]['State Name']
    state_with_highest_max_ph = avg_ph_by_state.loc[avg_ph_by_state['pH max'].idxmax()]['State Name']

    # Add annotations for lowest and highest average pH values
    fig.add_annotation(
        x=state_with_lowest_min_ph, y=avg_ph_by_state['pH min'].min(),
        text=f"Lowest Min pH ({avg_ph_by_state['pH min'].min():.2f})",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    fig.add_annotation(
        x=state_with_highest_max_ph, y=avg_ph_by_state['pH max'].max(),
        text=f"Highest Max pH ({avg_ph_by_state['pH max'].max():.2f})",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    sentences = [
        "It’s a bar graph with two sets of bars for each state, one indicating minimum (light purple) and another indicating maximum (dark purple) water pH levels.",
        "There are annotations for the highest max pH (13.64) and lowest min pH (5.91).",
        "The y-axis is labeled “pH” with a scale from 0 to 15.",
        "The x-axis is labeled “State” with names of various states listed."
    ]
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), 'State-wise Average Minimum and Maximum Water pH', sentences]

def StateWiseWaterTemperature():
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))

    # Calculate average min and max temperature by state
    avg_temp_by_state = df.groupby('State Name').agg({
        'Temperature (Celsius) min': 'mean',
        'Temperature (Celsius) max': 'mean'
    }).reset_index()

    # Plotting using Plotly with custom colors for min and max bars
    fig = px.bar(avg_temp_by_state, x='State Name', y=['Temperature (Celsius) min', 'Temperature (Celsius) max'],
                barmode='group', labels={'parameter': 'Temperature (Celsius)'}, 
                title='State-wise Average Minimum and Maximum Water Temperature')

    # Define custom colors for the bars (min: blue, max: red)
    color_map = {'Temperature (Celsius) min': 'skyblue', 'Temperature (Celsius) max': 'orange'}

    # Update layout and color mapping for better visualization
    fig.update_layout( xaxis_title='State',yaxis_title='Temperature °C', legend_title='Temperature Type',
        width=1400,  # Set width of the plot (adjust as needed)
        height=700,  # Set height of the plot (adjust as needed)
        margin=dict(l=60, r=60, t=90, b=110)  # Adjust margin to provide space for labels
    )

    # Apply custom colors to the bars
    for trace, color in color_map.items():
        fig.update_traces(marker_color=color, selector=dict(name=trace))

    # Find states with the lowest and highest average temperatures
    state_with_lowest_min_temp = avg_temp_by_state.loc[avg_temp_by_state['Temperature (Celsius) min'].idxmin()]['State Name']
    state_with_highest_max_temp = avg_temp_by_state.loc[avg_temp_by_state['Temperature (Celsius) max'].idxmax()]['State Name']

    # Add annotations for lowest and highest average temperatures
    fig.add_annotation(
        x=state_with_lowest_min_temp, y=avg_temp_by_state['Temperature (Celsius) min'].min(),
        text=f"Lowest Min Temp ({avg_temp_by_state['Temperature (Celsius) min'].min():.2f}°C)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )

    fig.add_annotation(
        x=state_with_highest_max_temp, y=avg_temp_by_state['Temperature (Celsius) max'].max(),
        text=f"Highest Max Temp ({avg_temp_by_state['Temperature (Celsius) max'].max():.2f}°C)",
        showarrow=True, arrowhead=1, ax=0, ay=-40
    )
    sentences = ["The x-axis labels indicate different states, while the y-axis represents temperature in Celsius, ranging from 0 to 35°C.",
                 "Each state has two bars: blue for minimum temperature and orange for maximum temperature.",
                "The highest maximum temperature recorded is 32.42°C in West Bengal, while the lowest minimum is 0°C in Himachal Pradesh"
    ]
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), 'State-wise Average Minimum and Maximum Water Temperature', sentences]


def getPredictivemodel(modelTorender):
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))
    # Calculate Water Quality Index (WQI) for each row# Calculate Water Quality Index (WQI) for each row
    df['WQI'] = df.apply(calculate_wqi, axis=1)

    # Define the desired range for scaled WQI (0 to 400)
    min_range = 0
    max_range = 400

    # Scale the WQI values to the specified range
    min_wqi = df['WQI'].min()
    max_wqi = df['WQI'].max()

    df['Scaled WQI'] = ((df['WQI'] - min_wqi) / (max_wqi - min_wqi)) * (max_range - min_range) + min_range

    # Display the scaled WQI values
    print("Scaled WQI:")
    print(df['Scaled WQI'])

    # Feature Selection and Data Splitting
    X = df[numerical_columns]  # Use numerical columns for model training
    y = df['Scaled WQI']               # WQI is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training and Evaluation
    models = {
        'Random Forest': RandomForestRegressor(),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Multi-Linear Regression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred).round(5)
        rmse = np.sqrt(mse).round(5)
        mae = mean_absolute_error(y_test, y_pred).round(5)
        r2 = r2_score(y_test, y_pred).round(5)
        
        results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R-squared': r2}

    # Generate and Display Plots
    metric_names = ['MAE', 'RMSE', 'MSE', 'R-squared']

    # Define colors for each model
    colors = ['rgba(31, 119, 180, 0.7)', 'rgba(255, 127, 14, 0.7)', 'rgba(44, 160, 44, 0.7)', 'rgba(214, 39, 40, 0.7)', 'rgba(148, 103, 189, 0.7)']

    # Plot 1: Mean Absolute Error (MAE)
    mae_data = {'Model': list(results.keys()), modelTorender: [results[model][modelTorender] for model in results]}
    mae_summary_df = pd.DataFrame(mae_data, index=range(1, len(mae_data['Model']) + 1))
    # Plot MAE values

    HeadTitle = ''
    yTitle = ''
    sentences = []
    if modelTorender == "MAE":
        HeadTitle = 'Mean Absolute Error (MAE) of Prediction Models'
        yTitle = 'MAE Value'
        sentences = [
            "The x-axis represents different prediction “Model” types: Random Forest, Support Vector Machine, K-Nearest Neighbours, Decision Tree, and Multi-Linear Regression.",
            "The y-axis represents the “MAE Value” ranging from 0 to 120.",
            "Each bar corresponds to a model and has a distinct colour",
            "Blue for Random Forest (MAE slightly above 0)",
            "Orange for Support Vector Machine (MAE just above 100)",
            "Green for K-Nearest Neighbours (MAE around 70)"
            "Light blue for Decision Tree (almost zero MAE)",
            "Purple for Multi-Linear Regression (also almost zero MAE)",
            "Annotations indicate the highest max MAE (Support Vector Machine) and lowest min MAE (Decision Tree)",
            "The graph helps us compare the accuracy of different prediction models.",
	        "Models with lower MAE values are more accurate in their predictions."
            ]
    elif modelTorender == 'RMSE':
        HeadTitle = 'Root Mean Squared Error (RMSE) of Prediction Models'
        yTitle = 'RMSE Value'
        sentences = [
            "The x-axis represents five different machine learning models: Random Forest, Support Vector Machine, K-Nearest Neighbors, Decision Tree, and Multi-Linear Regression.",
            "The y-axis shows the Root Mean Squared Error (RMSE) values. RMSE measures how well a model’s predictions match the actual data. Lower RMSE values indicate better performance.",
            "Support Vector Machine (SVM) has the highest error, as indicated by the tallest bar.",
            "K-Nearest Neighbors (KNN) follows with a slightly lower error.",
            "Multi-Linear Regression has a moderate error.",
            "Decision Tree performs better than the previous models.",
            "Random Forest has the lowest error, represented by the shortest bar",
            "In summary, Random Forest is the best-performing model in terms of RMSE, while Support Vector Machine performs the worst. Lower RMSE values are desirable for accurate predictions."
        ]
    elif modelTorender == "MSE":
        HeadTitle = 'Mean Squared Error (MSE) of Prediction Models'
        yTitle = 'MSE Value'
        sentences = [
            "The x-axis represents five different machine learning models: Random Forest, Support Vector Machine, K-Nearest Neighbors, Decision Tree, and Multi-Linear Regression.",
            "The y-axis shows the Mean Squared Error (MSE) values. MSE measures how well a model’s predictions match the actual data. Lower MSE values indicate better performance.",
            "Support Vector Machine (SVM) has the highest error, as indicated by the tallest bar.",
            "K-Nearest Neighbors (KNN) follows with a slightly lower error.",
            "Multi-Linear Regression has a moderate error.",
            "Decision Tree performs better than the previous models.",
            "Random Forest has the lowest error, represented by the shortest bar"
            "In summary, Random Forest is the best-performing model in terms of MSE, while Support Vector Machine performs the worst.  lower MSE values are desirable for accurate predictions."
        ]
    else:
        HeadTitle = 'R-squared (R2) of Prediction Models'
        yTitle = 'R-squared Value'
        sentences = [
            "The x-axis represents five different machine learning models: Random Forest, Support Vector Machine, K-Nearest Neighbours, Decision Tree, and Multi-Linear Regression.",
            "The y-axis shows the R-squared (R2) values. R2 measures how well a model’s predictions fit the actual data. It ranges from 0 to 1, where 1 indicates a perfect fit.",
            "Random Forest has the highest R-squared value, almost 1. This suggests that Random Forest is an effective model according to this metric.",
            "Decision Tree also shows a high R-squared value, slightly less than Random Forest.",
            "Multi-Linear Regression and K-Nearest Neighbours have significantly lower R-squared values compared to Random Forest and Decision Tree.",
            "Support Vector Machine has an extremely low R-squared value close to zero, indicating it is not effective according to this metric.",
            "In summary, Random Forest performs best in terms of R-squared, while Support Vector Machine performs poorly. Higher R-squared values indicate better model fit."
        ]

    mae_fig = go.Figure(go.Bar(x=mae_summary_df['Model'], y=mae_summary_df[modelTorender], name=modelTorender, marker_color=colors))
    mae_fig.update_layout(title=HeadTitle,
                        xaxis_title='Model',
                        yaxis_title=yTitle)
    
    print("mae_summary_df",mae_summary_df)
    return [mae_fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), HeadTitle, sentences,mae_summary_df, modelTorender]


def ComparativeAnalysisOfWaterModel():
    df = pd.read_excel(f"xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))


    # Calculate Water Quality Index (WQI) for each row
    df['WQI'] = df.apply(calculate_wqi, axis=1)

    # Define the desired range for scaled WQI (0 to 400)
    min_range = 0
    max_range = 400

    # Scale the WQI values to the specified range
    min_wqi = df['WQI'].min()
    max_wqi = df['WQI'].max()

    df['Scaled WQI'] = ((df['WQI'] - min_wqi) / (max_wqi - min_wqi)) * (max_range - min_range) + min_range

    # Display the scaled WQI values
    print("Scaled WQI:")
    print(df['Scaled WQI'])

    # Feature Selection and Data Splitting
    X = df[numerical_columns]  # Use numerical columns for model training
    y = df['Scaled WQI']               # WQI is the target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training and Evaluation
    models = {
        'Random Forest': RandomForestRegressor(),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Multi-Linear Regression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred).round(5)
        rmse = np.sqrt(mse).round(5)
        mae = mean_absolute_error(y_test, y_pred).round(5)
        r2 = r2_score(y_test, y_pred).round(5)
        
        results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R-squared': r2}


    metric_names = ['MAE', 'RMSE', 'MSE', 'R-squared']
    best_model_criteria = {'MAE': min,'RMSE': min,'MSE': min,'R-squared': max}
    fig = make_subplots(rows=2, cols=2, subplot_titles=metric_names)

    colors = ['rgba(31, 119, 180, 0.7)', 'rgba(255, 127, 14, 0.7)', 'rgba(44, 160, 44, 0.7)', 'rgba(214, 39, 40, 0.7)', 'rgba(148, 103, 189, 0.7)']

    for i, metric in enumerate(metric_names):
        # Determine best model based on the metric (lower is better for MAE, RMSE, MSE; higher is better for R-squared)
        best_model = best_model_criteria[metric](results, key=lambda x: results[x][metric])
        
        for j, (model, color) in enumerate(zip(results.keys(), colors)):

            fig.add_trace(go.Bar(x=[model.replace(" ", "<br>")], y=[results[model][metric]], name=model, marker_color=color),
                        row=(i // 2) + 1, col=(i % 2) + 1)

            if model == best_model:
                best_value = results[model][metric]
                fig.add_annotation(x=model.replace(" ", "<br>"), y=best_value, text='Best', showarrow=True,
                                arrowhead=1, ax=0, ay=-40, row=(i // 2) + 1, col=(i % 2) + 1)

    fig.update_layout(height=800, width=1440, title_text="Comparative Performance of Prediction Models", showlegend=False)
    sentences = [
        "The x-axis represents five different machine learning models: Random Forest, Support Vector Machine, K-Nearest Neighbors, Decision Tree, and Multi-Linear Regression.",
        "MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual values. Lower MAE values indicate better performance.",
        "RMSE (Root Mean Square Error): Similar to MAE but emphasizes larger errors. RMSE is the square root of the average squared differences.",
        "MSE (Mean Squared Error): Measures the average squared difference between predicted and actual values. Lower MSE values are desirable.",
        "R-squared (R2): Indicates how well the model fits the data. R2 ranges from 0 to 1, with higher values indicating better fit.",
        "Random Forest performs best in terms of MAE, RMSE, and MSE. It has the lowest error across these metrics.",
        "Support Vector Machine excels in RMSE, showing the lowest error for this metric.",
        "Decision Tree also performs well in RMSE and MSE.",
        "Multi-Linear Regression and K-Nearest Neighbors have higher errors across all metrics",
        "Random Forest leads in R-squared, indicating better fit to the data",
        "In summary, Random Forest is the most effective model overall, while Support Vector Machine performs well in RMSE."
    ]

    metric_df = pd.DataFrame(columns=['Model'] + metric_names)

    for model, metrics in results.items():
        metric_values = [metrics[metric] for metric in metric_names]
        metric_df.loc[len(metric_df)+1] = [model] + metric_values

    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), "Comparative Performance of Prediction Models", sentences,'', '',metric_df]

def DiurnalLimitsTrendOfNoise():
    df = pd.read_excel("xlsxFiles/NOISE/noise_pollution_dataset.xlsx")

    # Convert 'Year', 'Month', 'DayLimit', and 'NightLimit' columns to datetime
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))

    # Group by month and calculate mean day and night limits
    monthly_mean = df.groupby('Date').agg({'DayLimit': 'mean', 'NightLimit': 'mean'}).reset_index()

    # Plotting with Plotly
    trace1 = go.Scatter(x=monthly_mean['Date'], y=monthly_mean['DayLimit'], mode='lines', name='Day Limit')
    trace2 = go.Scatter(x=monthly_mean['Date'], y=monthly_mean['NightLimit'], mode='lines', name='Night Limit')

    layout = go.Layout(title='Yearly Trends of Day and Night Limits',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Limit'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    sentences = []
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), 'Yearly Trends of Day and Night Limits', sentences]

def disparityAcrossAreaForNoise():
    df = pd.read_excel("xlsxFiles/NOISE/noise_pollution_dataset.xlsx")

    # Group by Type (Residential, Commercial, Industrial, Silence) and calculate mean day and night limits
    type_mean = df.groupby('Type').agg({'DayLimit': 'mean', 'NightLimit': 'mean'}).reset_index()

    # Plotting with Plotly
    trace1 = go.Bar(x=type_mean['Type'], y=type_mean['DayLimit'], name='Day Limit', marker=dict(color='blue'))
    trace2 = go.Bar(x=type_mean['Type'], y=type_mean['NightLimit'], name='Night Limit', marker=dict(color='red'))

    layout = go.Layout(title='Comparison of Day and Night Limits between Different Types of Areas',
                    xaxis=dict(title='Area Type'),
                    yaxis=dict(title='Limit'),
                    barmode='group')
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    sentences = []
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), 'Comparison of Day and Night Limits between Different Types of Areas', sentences]


def GetPredictiveModelOfNoise(modelTorender):
    # Load the dataset
    data = pd.read_excel("xlsxFiles/NOISE/noise_pollution_dataset.xlsx")
    data.dropna(subset=['Day', 'Night'], inplace=True)
    # Prepare the data
    X = data.drop(columns=["Station", "Name", "City", "State", "Type", "Year", "Month", "Day", "Night"])
    y_day = data["Day"]
    y_night = data["Night"]

    # Split the data into training and testing sets
    X_train, X_test, y_train_day, y_test_day, y_train_night, y_test_night = train_test_split(X, y_day, y_night, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Decision Trees": DecisionTreeRegressor(),
        "Multi-Linear Regression": LinearRegression()
    }

    # Create lists to store results
    results = []

    # Train and evaluate models
    for name, model in models.items():
        # Train the model for daytime
        model.fit(X_train, y_train_day)
        
        # Make predictions for daytime
        y_pred_day = model.predict(X_test)
        
        # Evaluate the model for daytime
        mse_day = mean_squared_error(y_test_day, y_pred_day)
        rmse_day = sqrt(mse_day)
        mae_day = mean_absolute_error(y_test_day, y_pred_day)
        r2_day = r2_score(y_test_day, y_pred_day)
        
        # Train the model for nighttime
        model.fit(X_train, y_train_night)
        y_pred_night = model.predict(X_test)
        
        # Evaluate the model for nighttime
        mse_night = mean_squared_error(y_test_night, y_pred_night)
        rmse_night = sqrt(mse_night)
        mae_night = mean_absolute_error(y_test_night, y_pred_night)
        r2_night = r2_score(y_test_night, y_pred_night)
        
        # Append results to the list
        results.append({
            "Model": name,
            "Daytime MSE": mse_day,
            "Daytime RMSE": rmse_day,
            "Daytime MAE": mae_day,
            "Daytime R-squared": r2_day,
            "Nighttime MSE": mse_night,
            "Nighttime RMSE": rmse_night,
            "Nighttime MAE": mae_night,
            "Nighttime R-squared": r2_night
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    HeadTitle = ''
    if modelTorender == "MAE":
        HeadTitle = 'Mean Absolute Error (MAE) of Prediction Models'
    elif modelTorender == 'RMSE':
        HeadTitle = 'Root Mean Squared Error (RMSE) of Prediction Models'
    elif modelTorender == "MSE":
        HeadTitle = 'Mean Squared Error (MSE) of Prediction Models'
    else:
        HeadTitle = 'R-squared (R2) of Prediction Models'

    fig_rmse = go.Figure()
    for index, row in results_df.iterrows():
        show_legend_daytime = index == 0
        show_legend_nighttime = index == 0
        fig_rmse.add_trace(go.Bar(
            x=[row['Model']],
            y=[row[f'Daytime {modelTorender}']],
            name=f'Daytime {modelTorender}' if show_legend_daytime else '',
            marker_color='rgb(55, 83, 109)',
            showlegend=show_legend_daytime
        ))
        fig_rmse.add_trace(go.Bar(
            x=[row['Model']],
            y=[row[f'Nighttime {modelTorender}']],
            name=f'Nighttime {modelTorender}' if show_legend_nighttime else '',
            marker_color='rgb(26, 118, 255)',
            showlegend=show_legend_nighttime
        ))
    fig_rmse.update_layout(title=HeadTitle,
                        xaxis_title='Model',
                        yaxis_title=modelTorender,
                        barmode='group')
    sentences = []
    
    df  = results_df.loc[:, ["Model", f"Daytime {modelTorender}", f"Nighttime {modelTorender}"]]

    print(df)

    return [fig_rmse.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), HeadTitle, sentences,df, modelTorender, None, df]

def ComparativeAnalysisOfNoiseModel():
    data = pd.read_excel("xlsxFiles/NOISE/noise_pollution_dataset.xlsx")
    data.dropna(subset=['Day', 'Night'], inplace=True)

    # Prepare the data
    X = data.drop(columns=["Station", "Name", "City", "State", "Type", "Year", "Month", "Day", "Night"])
    y_day = data["Day"]
    y_night = data["Night"]

    # Split the data into training and testing sets
    X_train, X_test, y_train_day, y_test_day, y_train_night, y_test_night = train_test_split(X, y_day, y_night, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Decision Trees": DecisionTreeRegressor(),
        "Multi-Linear Regression": LinearRegression()
    }

    # Create lists to store results
    results = []

    # Train and evaluate models
    for name, model in models.items():
        # Train the model for daytime
        model.fit(X_train, y_train_day)
        
        # Make predictions for daytime
        y_pred_day = model.predict(X_test)
        
        # Evaluate the model for daytime
        mse_day = mean_squared_error(y_test_day, y_pred_day)
        rmse_day = sqrt(mse_day)
        mae_day = mean_absolute_error(y_test_day, y_pred_day)
        r2_day = r2_score(y_test_day, y_pred_day)
        
        # Train the model for nighttime
        model.fit(X_train, y_train_night)
        y_pred_night = model.predict(X_test)
        
        # Evaluate the model for nighttime
        mse_night = mean_squared_error(y_test_night, y_pred_night)
        rmse_night = sqrt(mse_night)
        mae_night = mean_absolute_error(y_test_night, y_pred_night)
        r2_night = r2_score(y_test_night, y_pred_night)
        
        # Calculate mean metrics
        mean_rmse = (rmse_day + rmse_night) / 2
        mean_mae = (mae_day + mae_night) / 2
        mean_mse = (mse_day + mse_night) / 2
        mean_r2 = (r2_day + r2_night) / 2
        
        # Append results to the list
        results.append({
            "Model": name,
            "Mean RMSE": mean_rmse,
            "Mean MAE": mean_mae,
            "Mean MSE": mean_mse,
            "Mean R-squared": mean_r2
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Find the best model for each metric
    best_rmse_model = results_df.loc[results_df['Mean RMSE'].idxmin()]
    best_mae_model = results_df.loc[results_df['Mean MAE'].idxmin()]
    best_mse_model = results_df.loc[results_df['Mean MSE'].idxmin()]
    best_r2_model = results_df.loc[results_df['Mean R-squared'].idxmax()]

    # Define different colors for bars
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)']

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Mean RMSE", "Mean MAE", "Mean MSE", "Mean R-squared"))

    # Add traces
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Mean RMSE'], marker_color=colors, name='Mean RMSE', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Mean MAE'], marker_color=colors, name='Mean MAE', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Mean MSE'], marker_color=colors, name='Mean MSE', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Mean R-squared'], marker_color=colors, name='Mean R-squared', showlegend=False),
        row=2, col=2
    )

    # Highlight best models
    fig.add_annotation(
        x=best_rmse_model['Model'], y=best_rmse_model['Mean RMSE'], text='Best Model', showarrow=True, arrowhead=2, arrowcolor='red', arrowwidth=2, ax=0, ay=-40,
        row=1, col=1
    )
    fig.add_annotation(
        x=best_mae_model['Model'], y=best_mae_model['Mean MAE'], text='Best Model', showarrow=True, arrowhead=2, arrowcolor='red', arrowwidth=2, ax=0, ay=-40,
        row=1, col=2
    )
    fig.add_annotation(
        x=best_mse_model['Model'], y=best_mse_model['Mean MSE'], text='Best Model', showarrow=True, arrowhead=2, arrowcolor='red', arrowwidth=2, ax=0, ay=-40,
        row=2, col=1
    )
    fig.add_annotation(
        x=best_r2_model['Model'], y=best_r2_model['Mean R-squared'], text='Best Model', showarrow=True, arrowhead=2, arrowcolor='red', arrowwidth=2, ax=0, ay=-40,
        row=2, col=2
    )

    # Update layout
    fig.update_layout(height=800, width=1400, title_text="Model Performance Comparison")
    sentences = []
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), "Comparative Performance of Prediction Models", sentences,'', '',None]

def getWaterDataSet():
    df = pd.read_excel("xlsxFiles/WATER/water_quality_of_ground_water_state_wise_2019.xlsx")

    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col != 'Station Code' and col!= 'Year']
    #print("\n Numerical columns present in datasets (excluding the station code column) are as follows : \n ", numerical_columns)

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 1))

    return df

def ComparativeAnalysisOfAirModel():
    data = pd.read_excel("xlsxFiles/AIR/2013.xlsx")
    # Basic data cleaning (handle missing values, etc.)
    # Separate numeric columns from non-numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns

    # Fill missing values in numeric columns with the mean of each column
    data[numeric_columns] = data[numeric_columns].apply(lambda x: x.fillna(x.mean()))

    # Optionally, handle missing values in non-numeric columns (e.g., fill with mode, forward fill, etc.)
    # For demonstration, let's fill with the mode (most frequent value)
    data[non_numeric_columns] = data[non_numeric_columns].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))


    pd.set_option('display.max_rows', None)
    print(data)
    # Define features and target variables
    features = data.drop(['Annual Average SO2', 'Annual Average NO2', 'Annual Average PM10'], axis=1)  # Input features
    target_SO2 = data['Annual Average SO2']  # Target variable for SO2
    target_NO2 = data['Annual Average NO2']  # Target variable for NO2
    target_PM10 = data['Annual Average PM10']  # Target variable for PM10

    # Handling categorical columns with one-hot encoding
    features_encoded = pd.get_dummies(features)

    # Split data into training and testing sets using the encoded features
    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(features_encoded, target_SO2, test_size=0.2, random_state=42)

    # Initialize models
    rf_model = RandomForestRegressor()
    svm_model = SVR()
    knn_model = KNeighborsRegressor()
    dt_model = DecisionTreeRegressor()
    lr_model = LinearRegression()

    # Train the models with encoded features
    rf_model.fit(X_train_encoded, y_train)
    svm_model.fit(X_train_encoded, y_train)
    knn_model.fit(X_train_encoded, y_train)
    dt_model.fit(X_train_encoded, y_train)
    lr_model.fit(X_train_encoded, y_train)

    # Predictions
    rf_pred = rf_model.predict(X_test_encoded)
    svm_pred = svm_model.predict(X_test_encoded)
    knn_pred = knn_model.predict(X_test_encoded)
    dt_pred = dt_model.predict(X_test_encoded)
    lr_pred = lr_model.predict(X_test_encoded)

    # Calculate evaluation metrics for each model
    mse_scores = [mean_squared_error(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]
    rmse_scores = [np.sqrt(mse) for mse in mse_scores]
    mae_scores = [mean_absolute_error(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]
    r2_scores = [r2_score(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]

    models = ['Random Forest', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'Linear Regression']
    # Data preparation
    metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']
    scores = [mse_scores, rmse_scores, mae_scores, r2_scores]

    # Define a color scale
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create subplots with optimized layout
    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics, vertical_spacing=0.2)  # Adjust the vertical spacing here

    # Determine the length of the models array
    num_models = len(models)

    # Add traces for each metric
    for i, metric in enumerate(metrics):
        color_indices = np.arange(num_models) % len(colors)  # Ensure color indices are within the bounds of the colors array
        row = i // 2 + 1  # Calculate row index
        col = i % 2 + 1   # Calculate column index
        trace = go.Bar(
            x=models,
            y=scores[i],
            name=metric,
            marker=dict(color=[colors[j] for j in color_indices])  # Use a different color for each bar
        )
        fig.add_trace(trace, row=row, col=col)

        # Highlight the best model for each metric
        best_model_index = np.argmin(scores[i]) if metric != 'R-squared' else np.argmax(scores[i])
        fig.add_annotation(
            x=models[best_model_index],
            y=scores[i][best_model_index],
            xref='x',
            yref='y',
            text=f'Best: {models[best_model_index]}',
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            row=row,
            col=col
        )

    # Update layout
    fig.update_layout(
        title='Metrics for Different Models',
        width=1400,
        height=800,
    )

    # Make ticks horizontal and wrap them
    ticktexts = [label.replace(' ', '<br>') for label in models]  # Insert <br> to represent space
    fig.update_xaxes(tickvals=np.arange(len(models)), ticktext=ticktexts)
    fig.update_yaxes(tickangle=0, title='Score')
    sentences = []
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), "Comparative Performance of Prediction Models", sentences,'', '',None]

def GetPredictiveModelOfAir(model):
    data = pd.read_excel("xlsxFiles/AIR/2013.xlsx")
    # Basic data cleaning (handle missing values, etc.)
    # Separate numeric columns from non-numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns

    # Fill missing values in numeric columns with the mean of each column
    data[numeric_columns] = data[numeric_columns].apply(lambda x: x.fillna(x.mean()))

    # Optionally, handle missing values in non-numeric columns (e.g., fill with mode, forward fill, etc.)
    # For demonstration, let's fill with the mode (most frequent value)
    data[non_numeric_columns] = data[non_numeric_columns].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))


    pd.set_option('display.max_rows', None)
    features = data.drop(['Annual Average SO2', 'Annual Average NO2', 'Annual Average PM10'], axis=1)  # Input features
    target_SO2 = data['Annual Average SO2']  # Target variable for SO2
    target_NO2 = data['Annual Average NO2']  # Target variable for NO2
    target_PM10 = data['Annual Average PM10']  # Target variable for PM10

    # Handling categorical columns with one-hot encoding
    features_encoded = pd.get_dummies(features)

    # Split data into training and testing sets using the encoded features
    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(features_encoded, target_SO2, test_size=0.2, random_state=42)

    # Initialize models
    rf_model = RandomForestRegressor()
    svm_model = SVR()
    knn_model = KNeighborsRegressor()
    dt_model = DecisionTreeRegressor()
    lr_model = LinearRegression()

    # Train the models with encoded features
    rf_model.fit(X_train_encoded, y_train)
    svm_model.fit(X_train_encoded, y_train)
    knn_model.fit(X_train_encoded, y_train)
    dt_model.fit(X_train_encoded, y_train)
    lr_model.fit(X_train_encoded, y_train)

    # Predictions
    rf_pred = rf_model.predict(X_test_encoded)
    svm_pred = svm_model.predict(X_test_encoded)
    knn_pred = knn_model.predict(X_test_encoded)
    dt_pred = dt_model.predict(X_test_encoded)
    lr_pred = lr_model.predict(X_test_encoded)

    # Calculate evaluation metrics for each model
    mse_scores = [mean_squared_error(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]
    rmse_scores = [np.sqrt(mse) for mse in mse_scores]
    mae_scores = [mean_absolute_error(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]
    r2_scores = [r2_score(y_test, pred) for pred in [rf_pred, svm_pred, knn_pred, dt_pred, lr_pred]]

    models = ['Random Forest', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'Linear Regression']
    # Data preparation
    metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']
    # scores = [mse_scores, rmse_scores, mae_scores, r2_scores]

    # metric = ''
    # I = 0
    # if model == "MSE":
    #     metric = "MSE"
    #     I=0
    # elif model == "RMSE": 
    #     metric = "RMSE" 
    #     I=1
    # elif model == "MAE":
    #     metric = "MAE"
    #     I=2
    # else:
    #     metric = "R-squared"
    #     I=3

    fig = {}

    evaluation_df = pd.DataFrame({'Model': models,
                              'Mean Squared Error': mse_scores,
                              'Root Mean Squared Error': rmse_scores,
                              'Mean Absolute Error': mae_scores,
                              'R-squared': r2_scores})

    # Define a color sequence for bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    HeadTitle = ''
    if model == "MSE":
        mse_plot = go.Bar(x=evaluation_df['Model'], y=evaluation_df['Mean Squared Error'], name='Mean Squared Error', marker=dict(color=colors))
        mse_layout = go.Layout(title='Mean Squared Error for Different Models', xaxis=dict(tickangle=-45, showgrid=False), yaxis=dict(showgrid=False), width=800, height=500)
        mse_fig = go.Figure(data=[mse_plot], layout=mse_layout)
        fig = mse_fig
        HeadTitle = 'Mean Squared Error for Different Models'
    elif model == "RMSE": 
        rmse_plot = go.Bar(x=evaluation_df['Model'], y=evaluation_df['Root Mean Squared Error'], name='Root Mean Squared Error', marker=dict(color=colors))
        rmse_layout = go.Layout(title='Root Mean Squared Error for Different Models', xaxis=dict(tickangle=-45, showgrid=False), yaxis=dict(showgrid=False), width=800, height=500)
        rmse_fig = go.Figure(data=[rmse_plot], layout=rmse_layout)
        fig = rmse_fig
        HeadTitle = 'Root Mean Squared Error for Different Models'
    elif model == "MAE":
        mae_plot = go.Bar(x=evaluation_df['Model'], y=evaluation_df['Mean Absolute Error'], name='Mean Absolute Error', marker=dict(color=colors))
        mae_layout = go.Layout(title='Mean Absolute Error for Different Models', xaxis=dict(tickangle=-45, showgrid=False), yaxis=dict(showgrid=False), width=800, height=500)
        mae_fig = go.Figure(data=[mae_plot], layout=mae_layout)
        fig = mae_fig
        HeadTitle = 'Mean Absolute Error for Different Models'
    else:
        r2_plot = go.Bar(x=evaluation_df['Model'], y=evaluation_df['R-squared'], name='R-squared', marker=dict(color=colors))
        r2_layout = go.Layout(title='R-squared for Different Models', xaxis=dict(tickangle=-45, showgrid=False), yaxis=dict(showgrid=False), width=800, height=500)
        r2_fig = go.Figure(data=[r2_plot], layout=r2_layout)
        fig = r2_fig
        HeadTitle = 'R-squared for Different Models'
    sentences = []
    df = None
    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), HeadTitle, sentences,df, model, None, df]