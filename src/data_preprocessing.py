import pandas as pd
from src.data_ingestion import  DataIngestion

class DataPreprocessing:
    def __init__(self,df:pd.DataFrame) -> None:
        self.data=df
        self.label2id=dict()
        self.id2label=dict()


    def remove_specific_entities(self)->pd.DataFrame:
        entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
        self.data= self.data[~self.data.Tag.isin(entities_to_remove)]
        return self.data

    def fill_empty_values(self)->pd.DataFrame:
        self.data = self.data.fillna(method='ffill')
        return self.data

    def create_new_columns(self)->pd.DataFrame:
        self.data["Sentence"]=self.data[["Sentence #","Word","Tag"]].groupby(["Sentence #"])["Word"].transform(lambda x:" ".join(x))
        self.data["Word_Labels"]=self.data[["Sentence #","Word","Tag"]].groupby(["Sentence #"])["Tag"].transform(lambda x:",".join(x))
        return self.data

    def create_dict_of_label_id_pairs(self):
        self.label2id = {k: v for v, k in enumerate(self.data.Tag.unique())}
        self.id2label = {v: k for v, k in enumerate(self.data.Tag.unique())}
        #print(self.label2id)

    def drop_duplicates(self)->pd.DataFrame:
        self.data=self.data[["Sentence","Word_Labels"]].drop_duplicates().reset_index(drop=True)
        return self.data
