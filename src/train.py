from from_root.root import from_root
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertConfig,BertForTokenClassification
from torch import cuda
from src.data_preprocessing import DataPreprocessing
from src.model_architecture import ModelArchitecture
from src.utils import UtilsForProgram
from src.data_ingestion import DataIngestion
from src.evaluation import EvaluateModel

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
MAX_GRAD_NORM = 10
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
EPOCHS = 1

class TrainModel:
    def __init__(self,training_set,testing_set,preprocessed_data:DataPreprocessing) -> None:
        self.train_params={'batch_size':TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
        self.test_params={'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
        self.model=None
        self.training_set=training_set
        self.preprocessed_data=preprocessed_data
        self.training_loader=DataLoader(training_set,**self.train_params)
        self.testing_loader=DataLoader(testing_set,**self.test_params)


    def define_model(self):
        self.model=BertForTokenClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=len(self.preprocessed_data.id2label),
                                                              id2label=self.preprocessed_data.id2label,
                                                              label2id=self.preprocessed_data.label2id)
        if torch.cuda.device_count() > 1:
            print(f"lets use { torch.cuda.device_count()} GPUs")
            self.model=torch.nn.DataParallel(self.model)
        self.model.to(UtilsForProgram.device)
        self.optimizer=torch.optim.Adam(params=self.model.parameters(),lr=LEARNING_RATE)
        #print(self.model)

    def verify_loss_before_training(self):
        ids = self.training_set[0]["ids"].unsqueeze(0)
        mask = self.training_set[0]["mask"].unsqueeze(0)
        targets = self.training_set[0]["targets"].unsqueeze(0)
        ids = ids.to(UtilsForProgram.device)
        mask = mask.to(UtilsForProgram.device)
        targets = targets.to(UtilsForProgram.device)
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
        initial_loss = outputs[0]
        print(initial_loss)
        #Get the output shape -verify that the logits of the neural network have a shape of (batch_size, sequence_length, num_labels):
        tr_logits = outputs[1]
        print(tr_logits.shape)
        for epoch in range(EPOCHS):
            print(f"Training epoch: {epoch + 1}")
            self.train(epoch)
        return self.model

    # Defining the training function on the 80% of the dataset for tuning the bert model
    def train(self,epoch):
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        self.model.train()

        for idx, batch in enumerate(self.training_loader):

            ids = batch['ids'].to(UtilsForProgram.device, dtype = torch.long)
            mask = batch['mask'].to(UtilsForProgram.device, dtype = torch.long)
            targets = batch['targets'].to(UtilsForProgram.device, dtype = torch.long)

            outputs = self.model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, tr_logits = outputs.loss, outputs.logits
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")

            # compute training accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_preds.extend(predictions)
            tr_labels.extend(targets)

            tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")
        torch.save(self.model.state_dict(),from_root("artifacts","model","ner.pt"))

    def evaluate(self):
        self.model.load_state_dict(torch.load(from_root("artifacts","model","ner.pt")))
        evaluateModel=EvaluateModel(self.model,self.testing_loader,self.preprocessed_data.id2label)
        labels, predictions =evaluateModel.valid()
        return labels, predictions
