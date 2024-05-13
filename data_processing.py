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
    sentences = []
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
    sentences = []
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

    sentences = []
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
    sentences = []
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
    if modelTorender == "MAE":
        HeadTitle = 'Mean Absolute Error (MAE) of Prediction Models'
        yTitle = 'MAE Value'
    elif modelTorender == 'RMSE':
        HeadTitle = 'Root Mean Squared Error (RMSE) of Prediction Models'
        yTitle = 'RMSE Value'
    elif modelTorender == "MSE":
        HeadTitle = 'Mean Squared Error (MSE) of Prediction Models'
        yTitle = 'MSE Value'
    else:
        HeadTitle = 'R-squared (R2) of Prediction Models'
        yTitle = 'R-squared Value'

    mae_fig = go.Figure(go.Bar(x=mae_summary_df['Model'], y=mae_summary_df[modelTorender], name=modelTorender, marker_color=colors))
    mae_fig.update_layout(title=HeadTitle,
                        xaxis_title='Model',
                        yaxis_title=yTitle)
    
    print("mae_summary_df",mae_summary_df)
    sentences = []
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
    sentences = []

    metric_df = pd.DataFrame(columns=['Model'] + metric_names)

    for model, metrics in results.items():
        metric_values = [metrics[metric] for metric in metric_names]
        metric_df.loc[len(metric_df)+1] = [model] + metric_values

    return [fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500), "Comparative Performance of Prediction Models", sentences,'', '',metric_df]