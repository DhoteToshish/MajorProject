import pandas as pd

def load_data():
    data = pd.read_excel("xlsxFiles/2013_Data.xlsx")
    numeric_mean = data.mean(numeric_only=True)
    data.fillna(numeric_mean, inplace=True)
    pd.set_option('display.max_rows', None)
    return data
