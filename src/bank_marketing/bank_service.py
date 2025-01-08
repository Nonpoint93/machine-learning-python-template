import pandas as pd
import sys

class BankService:
    def __init__(self):
        pass


    def read_dataset(self) -> pd.DataFrame:

        """ Read a dataset """
        dataframe = None
        try:
            dataframe = pd.read_csv("data/raw/bank-marketing-campaign-data.csv", delimiter=";")
        except Exception as e:
            print(e)
            sys.exit(1)
        return dataframe
    
    def display_info_dataset(self, dataframe: pd.DataFrame):
        
        """ Print info from dataset """
        
        print(dataframe.info())
        print(dataframe.head())
        print(dataframe.shape)


    def process_data(self):
        dataframe = self.read_dataset()
        self.display_info_dataset(dataframe=dataframe)
