from keras.callbacks import EarlyStopping

def initialize():
    settings = {  
        'embedding_size': 10,   
        'axes' : 1,
        'normalize' : False,   
        'input_length' : 1,
        'shape' : [1]
    } 
    return settings

def fit():
    settings = {    
        'batch_size': 64, 
        'epochs': 5, 
        'validation_split': 0.1, 
        'verbose' : True,
        'shuffle' : True
        #'callbacks' : [EarlyStopping(monitor='val_loss',mode='min',verbose=0)]     
    } 
    return settings

def compile():
    settings = {
        'loss' : 'mse',
        'optimizer' : 'adam'
    }
    return settings 
