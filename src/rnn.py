import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, RepeatVector, Activation, TimeDistributed
from tensorflow.python.keras.optimizers import RMSprop

def batch_generator(batch_size, sequence_length, x , y):
    '''バッチサイズにデータを変換する関数
        args:
            batch_size (int): 隠れ層の数
            sequence_length (int): バッチサイズのデータ長
            x (numpy)      : バッチサイズに切り取った説明変数
            y (numpy)      : バッチサイズに切り取った目的変数
        
    '''

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, x.shape[1])
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, y.shape[1])
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            np.random.seed(0)
            idx = np.random.randint(len(x) - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x[idx:idx+sequence_length]
            y_batch[i] = y[idx:idx+sequence_length]
        yield (x_batch, y_batch)
        
def bi_directional_RNN(HIDDEN_SIZE, loss, optimizer, num_x, num_y, activation='tanh'):
    '''双方向RNNのモデル作成をする関数
        args:
            HIDDEN_SIZE (int): 隠れ層の数
            loss (keras.loss): loss関数
            optimizer (tensorflow.python.keras.optimizers)     : 誤差関数
            num_x (int)      : インプットするデータの次元数
            num_y (int)      : アウトプットするデータの次元数
            activation (string): 活性化関数名
        
    '''
    model = Sequential()
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), input_shape=(None, num_x)))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dense(num_y, activation = activation))
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())
    return model


def RNN(HIDDEN_SIZE, loss, optimizer, num_x, num_y, activation='tanh'):
    '''RNNのモデル作成をする関数
        args:
            HIDDEN_SIZE (int): 隠れ層の数
            loss (keras.loss): loss関数
            optimizer (tensorflow.python.keras.optimizers)     : 誤差関数
            num_x (int)      : インプットするデータの次元数
            num_y (int)      : アウトプットするデータの次元数
            activation (string): 活性化関数名
        
    '''
    model = Sequential()
    model.add(LSTM(units=HIDDEN_SIZE, return_sequences=True, input_shape=(None, num_x,)))
    model.add(Dense(num_y, activation = activation))
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())
    return model

def shape_train_test(data, train_split):
    '''学習データと検証データを分割する関数
        args:
            data (df): データ
            train_split (float): 学習データと検証データを分ける割合
        
    '''
    num_data = len(data)
    num_train = int(train_split * num_data)
    num_test = num_data - num_train
    train = data[0:num_train]
    test = data[num_train:]
    num_signals = data.shape[1]
    return num_signals, train, test