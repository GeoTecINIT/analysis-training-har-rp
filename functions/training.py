import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from timeit import default_timer as timer

tf.config.set_visible_devices([], 'GPU')

def create_model():
    keras.backend.clear_session()
    model = Sequential([
        Conv1D(filters=64, kernel_size=10, padding='same', activation='relu', input_shape=(6, 50)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

  
def create_trainer(batch_size, epochs):
    def train_model(x, y, validation_data=None, verbose=0):
        cb = TimingCallback()
        model = create_model()
        model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[cb], validation_data=validation_data, verbose=verbose)
        return model, sum(cb.logs)
    
    return train_model