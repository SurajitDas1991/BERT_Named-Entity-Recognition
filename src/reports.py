import pandas as pd
import numpy as np
from regex import D
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertConfig,BertForTokenClassification
from torch import cuda
from src.data_preprocessing import DataPreprocessing
from src.model_architecture import ModelArchitecture
from src.utils import UtilsForProgram
from src.data_ingestion import DataIngestion
from seqeval.metrics import classification_report


class ReportModel:
     def __init__(self) -> None:
         pass
     def show_reports(self,labels,predictions):
        print(classification_report([labels], [predictions]))
