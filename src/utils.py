from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# load the .env file variables
load_dotenv()

def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine



def remove_outliers(dataframe, column) -> pd.DataFrame:
    """ Eliminar outliers """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
