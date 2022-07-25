import pandas as pd
from from_root import from_root

class  DataIngestion:
    data=pd.DataFrame()
    @classmethod
    def get_dataframe_from_csv(cls):
        cls.data=pd.read_csv(from_root('data','ner_datasetreference.csv'),encoding='unicode_escape')
