import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertConfig,BertForTokenClassification
from torch import cuda
from src.data_preprocessing import DataPreprocessing
from src.evaluation import EvaluateModel
from src.inference import InferenceModel
from src.model_architecture import ModelArchitecture
from src.reports import ReportModel
from src.train import TrainModel
from src.utils import UtilsForProgram
from src.data_ingestion import DataIngestion


MAX_LEN=128

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():
    pass


def prepare_dataset(data:pd.DataFrame):
    train_size = 0.8
    train_dataset = data.sample(frac=train_size,random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    return train_dataset,test_dataset

def verify_input_ids_and_targets_match(training_set,preprocessed_data:DataPreprocessing):
    for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["ids"][:30]), training_set[0]["targets"][:30]):
        print('{0:10}  {1}'.format(token, preprocessed_data.id2label[label.item()]))


if __name__ == '__main__':
    UtilsForProgram.check_if_cuda_is_available()
    print(UtilsForProgram.device)
    DataIngestion.get_dataframe_from_csv()
    #print(DataIngestion.data.head())
    preprocessed_data= DataPreprocessing(DataIngestion.data)
    data=preprocessed_data.remove_specific_entities()
    data=preprocessed_data.fill_empty_values()
    data=preprocessed_data.create_new_columns()
    preprocessed_data.create_dict_of_label_id_pairs()
    data=preprocessed_data.drop_duplicates()
    train_dataset,test_dataset=prepare_dataset(data)
    training_set=ModelArchitecture(train_dataset,tokenizer,MAX_LEN,preprocessed_data.label2id)
    testing_set=ModelArchitecture(test_dataset,tokenizer,MAX_LEN,preprocessed_data.label2id)
    ###########Verification code###########
    # sentence_list,word_labels= UtilsForProgram.tokenize_and_preserve_labels(data.iloc[41].Sentence, data.iloc[41].Word_Labels, tokenizer)
    # for i in range(0,len(sentence_list)):
    #     print(f"{sentence_list[i]} is {word_labels[i]}")
    #######################################
    # print(data.head())
    # print(len(data))
    #print(training_set[0])
    #verify_input_ids_and_targets_match(training_set,preprocessed_data)
    trainModel=TrainModel(training_set,testing_set,preprocessed_data)
    trainModel.define_model()
    model=trainModel.verify_loss_before_training()
    labels, predictions=trainModel.evaluate()
    report=ReportModel()
    report.show_reports(labels, predictions)
    inference=InferenceModel(model,tokenizer,"India is a big country with places like New Delhi")
