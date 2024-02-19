import pandas as pd

def load_data(year):
    data = pd.read_excel(f"xlsxFiles/{year}_Data.xlsx")
    dataFrame = getDataFrame(data)
    return dataFrame


def getDataFrame(excleData):
    numeric_mean = excleData.mean(numeric_only=True)
    excleData.fillna(numeric_mean, inplace=True)
    pd.set_option('display.max_rows', None)
    dataFrame = excleData
    return dataFrame