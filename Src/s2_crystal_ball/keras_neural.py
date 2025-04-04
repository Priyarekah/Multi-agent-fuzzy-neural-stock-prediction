
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping
import pandas as pd 


def mlpConstructor(model_dict): 

    nn = Sequential()

    for layer in model_dict['mlp']['layers']: 

        nodes = model_dict['mlp']['layers'][layer]['nodes']
        activation = model_dict['mlp']['hl_activation']

        if layer == 0: nn.add(Dense(nodes, input_dim = model_dict['input_size'], activation = activation))
        else: nn.add(Dense(nodes, activation = activation))
    
    # add output layer 
    activation = model_dict['mlp']['ol_activation']
    nn.add(Dense(model_dict['output_size'], activation = activation))    
    
    # compile model 
    if activation == 'linear': nn.compile(loss = 'mse', optimizer = model_dict['mlp']['optimizer']['optim_type'])
    # elif activation == 'sigmoid': nn.compile(loss = 'binary_crossentropy', optimizer =  model_dict['mlp']['optimizer']['optim_type'], metrics = ['binary_accuracy'])
    # elif activation == 'softmax': nn.compile(loss = 'categorical_crossentropy', optimizer =  model_dict['mlp']['optimizer']['optim_type'], metrics = ['accuracy'])
    elif activation == 'sigmoid': nn.compile(loss = 'mse', optimizer =  model_dict['mlp']['optimizer']['optim_type'])
    elif activation == 'softmax': nn.compile(loss = 'categorical_crossentropy', optimizer =  model_dict['mlp']['optimizer']['optim_type'], metrics = ['accuracy'])


    return nn

# kerasTrain function 
# kerasTrain function
def kerasTrain(model, model_dict, cluster_details, trainset, valset, testset, header):
    from keras.callbacks import EarlyStopping

    # early stopper
    es = EarlyStopping(
        monitor='loss',
        mode='min',
        verbose=1,
        patience=model_dict['early_stopper']['patience'],
        min_delta=model_dict['early_stopper']['min_delta']
    )

    # fit model
    model.fit(
        trainset['X'], trainset['y'],
        epochs=model_dict['epochs'],
        batch_size=model_dict['batch_size'],
        shuffle=model_dict['mlp']['shuffle'],
        verbose=model_dict['mlp']['verbose'],
        validation_data=(valset['X'], valset['y']),
        callbacks=[es]
    )

    # predict
    train_pred = model.predict(trainset['X'], verbose=model_dict['mlp']['verbose'])
    val_pred = model.predict(valset['X'], verbose=model_dict['mlp']['verbose'])
    test_pred = model.predict(testset['X'], verbose=model_dict['mlp']['verbose'])

    # convert to DataFrames
    train_pred = pd.DataFrame(train_pred, columns=trainset['y'].columns, index=trainset['y'].index)
    val_pred = pd.DataFrame(val_pred, columns=valset['y'].columns, index=valset['y'].index)
    test_pred = pd.DataFrame(test_pred, columns=testset['y'].columns, index=testset['y'].index)

    return model, train_pred, val_pred, test_pred
