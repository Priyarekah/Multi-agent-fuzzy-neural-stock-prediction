
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import torch
import math
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, LBFGS, SGD, AdamW

def pytorchTrain(model, model_dict, trainset, valset, testset, traintype = 0):

    train_dataloader, trainref_dataloader, val_dataloader, test_dataloader = pytorchDataPreparation(model_dict, trainset, valset, testset)

    # retrieve variables
    n_epochs = model_dict['epochs']
    optimizer_choice = model_dict['optimizer'].get('optim_type', 'adam').lower()

    learning_rate = model_dict['optimizer'].get('learning_rate', 1e-3)
    weight_decay = model_dict['optimizer'].get('weight_decay', 0)

    patience = model_dict['early_stopper']['patience']
    min_delta = model_dict['early_stopper']['min_delta']

    # define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f' -- Connected to {device} -- ')

    # define loss function
    loss_fn_map = {
        "MSELoss": nn.MSELoss(),
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }

    loss_fn_name = model_dict.get('loss_fn', 'MSELoss')
    loss_fn = loss_fn_map.get(loss_fn_name, nn.MSELoss())


    # initialize early stopper
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    # define optimizer
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_choice}")

    history = []

    # optimal
    min_loss = math.inf

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in tqdm(range(n_epochs)):

        # set model to train mode
        model.train()

        train_batch = []

        for batch, (X_batch, y_batch) in enumerate(train_dataloader):

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # set gradients to zero before bacj
            optimizer.zero_grad()

            # forward pass
            if traintype == 1: train_output = model(X_batch, y_batch)
            else: train_output = model(X_batch)

            # calculate loss
            train_loss = loss_fn(train_output, y_batch)
            train_batch.append(train_loss.item())

            # backpropagation
            train_loss.backward() # calculate gradient descent
            optimizer.step() # update weights

        # evaluate model performance at end of epoch
        model.eval()

        train_loss = sum(train_batch)/len(train_batch)
        train_losses.append(train_loss)

        val_batch = []

        # disables gradient calculation
        with torch.no_grad():

            for batch, (X_reftrain, y_reftrain) in enumerate(trainref_dataloader): 

                X_reftrain, y_reftrain = X_reftrain.to(device), y_reftrain.to(device) 

                if traintype == 1: trainref_pred = model(X_reftrain, y_reftrain)
                else: trainref_pred = model(X_reftrain)

            for batch, (X_bval, y_bval) in enumerate(val_dataloader):
                # print(f'== {batch} debug here - val input')
                # print(X_bval)
                # print('=')
                # print(y_bval)


                X_bval, y_bval = X_bval.to(device), y_bval.to(device)

                if traintype == 1: val_pred = model(X_bval, y_bval)
                else: val_pred = model(X_bval)

                val_loss = loss_fn(val_pred, y_bval)

                val_batch.append(val_loss.item())

        val_loss = sum(val_batch)/len(val_batch)
        val_losses.append(val_loss)


        test_batch = []
        ref_epoch = 0
        min_loss = math.inf
        train_losses = []
        val_losses = []
        test_losses = []

        # disables gradient calculation
        with torch.no_grad():
            for batch, (X_btest, y_btest) in enumerate(test_dataloader):

                X_btest, y_btest = X_btest.to(device), y_btest.to(device)

                if traintype == 1: test_pred = model(X_btest, y_btest)
                else: test_pred = model(X_btest)

                test_loss = loss_fn(test_pred, y_btest)

                test_batch.append(test_loss.item())

        test_losses.append(sum(test_batch)/len(test_batch))
        
        if val_loss < min_loss:
            optimal_model = copy.deepcopy(model)
            optimal_train_pred = pd.DataFrame(trainref_pred.cpu(), columns = trainset['y'].columns, index = trainset['y'].index)
            optimal_val_pred = pd.DataFrame(val_pred.cpu(), columns = valset['y'].columns, index = valset['y'].index)
            optimal_test_pred = pd.DataFrame(test_pred.cpu(), columns = testset['y'].columns, index = testset['y'].index)
            min_loss = val_loss
            ref_epoch = epoch
        
        # early stop to prevent overfitting 
        if early_stopper.early_stop(val_loss): break

    print(f'Optimal Model Epoch: {ref_epoch}')

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1,len(val_losses)+1), val_losses,label='Validation Loss')
    # plt.plot(range(1,len(test_losses)+1), test_losses,label='Validation Loss')

    # find position of lowest validation loss
    minposs = test_losses.index(min(test_losses))
    # plt.axvline(ref_epoch, linestyle='--', color='r',label='Early Stopping (Val) Checkpoint')
    plt.axvline(ref_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')


    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return optimal_model, optimal_train_pred, optimal_val_pred, optimal_test_pred


def pytorchDataPreparation(model_dict, trainset, valset, testset):

    batch_size = model_dict['batch_size']
    shuffle = model_dict[model_dict['model_type']]['shuffle']

    train_data = FuzzyDataset(trainset['X'], trainset['y'])
    val_data = FuzzyDataset(valset['X'], valset['y'])
    test_data = FuzzyDataset(testset['X'], testset['y'])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    trainref_dataloader = DataLoader(train_data, batch_size=len(trainset['y']), shuffle = shuffle)
    val_dataloader = DataLoader(val_data, batch_size=len(valset['y']), shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=len(testset['y']), shuffle=shuffle)

    return train_dataloader, trainref_dataloader, val_dataloader, test_dataloader



class FuzzyDataset(Dataset):
    def __init__(self, X, y): 
        self.X = torch.tensor(X.to_numpy(), dtype = torch.float)
        self.y = torch.tensor(y.to_numpy(), dtype = torch.float)
    
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class Transformer(nn.Module):
    # Constructor
    def __init__(self, model_dict):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
    d_model=model_dict['transformer']['d_model'],
    nhead=model_dict['transformer']['nhead'],
    dim_feedforward=model_dict['transformer']['dim_feedforward'],
    dropout=model_dict['transformer']['dropout'],
    activation=model_dict['transformer']['activation'],
    batch_first=model_dict['transformer']['batch_first']
)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_dict['transformer']['num_encoder_layers'])
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
       
        self.hidden_size = model_dict['transformer']['hidden_size']
        self.dense_layers = model_dict['transformer']['dense_layers']
        
        self.ll1 = nn.Linear(model_dict['input_size'], self.hidden_size)
        self.ll2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ll3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ll4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ll5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ll6 = nn.Linear(self.hidden_size, model_dict['output_size'])
        
        if self.dense_layers == 1: self.ll1 = nn.Linear(model_dict['input_size'], model_dict['output_size'])
        elif self.dense_layers == 2: self.ll2 = nn.Linear(self.hidden_size, model_dict['output_size'])
        elif self.dense_layers == 3: self.ll3 = nn.Linear(self.hidden_size, model_dict['output_size'])
        elif self.dense_layers == 4: self.ll4 = nn.Linear(self.hidden_size, model_dict['output_size'])
        elif self.dense_layers == 5: self.ll5 = nn.Linear(self.hidden_size, model_dict['output_size'])
        

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.ll1(x)
        
        if self.dense_layers >= 2: 
            x = self.activation(x)
            x = self.ll2(x)
            
            if self.dense_layers >= 3:
                x = self.activation(x)
                x = self.ll3(x)
                
                if self.dense_layers >= 4: 
                    x = self.activation(x)
                    x = self.ll4(x)
                    
                    if self.dense_layers >= 5: 
                        x = self.activation(x)
                        x = self.ll5(x)
                        
                        if self.dense_layers >= 6:
                            x = self.activation(x)
                            x = self.ll6(x)

        x = self.output_activation(x)
        out = x

        return out

class MLP(nn.Module):
    def __init__(self, model_dict):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(model_dict['input_size'], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, model_dict['output_size']),            
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits