import numpy as np
from tensorflow import keras

def train_model(model,X_train, Y_train, X_test, Y_test,model_name):
    '''
    Function to train the model 
    Inputs: model
        train data, test data 
        model mame to save the model by the model name
    Output: History about the model
        Train/test accuracy, Train/Test loss.
    '''
    # Define callbacks

    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_name+"_alzheimer_model.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)

    # Train the model
    history = model.model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=10,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    return history

