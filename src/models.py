from keras.models import Sequential, Model
from keras.optimizers import *
from keras.layers import *

def rowColCNN():
    vanilla_cnn = Sequential()
    vanilla_cnn.add(Conv2D(28, (11, 11), input_shape = (224, 224, 3)))
    vanilla_cnn.add(BatchNormalization())
    vanilla_cnn.add(Activation('relu'))
    vanilla_cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    vanilla_cnn.add(Conv2D(28, (7, 7)))
    vanilla_cnn.add(BatchNormalization())
    vanilla_cnn.add(Activation('relu', name = 'vanilla_cnn_output'))
    
    
    row_cnn = Conv2D(56, (3, 56))(vanilla_cnn.get_layer('vanilla_cnn_output').output)
    row_cnn = BatchNormalization()(row_cnn)
    row_cnn = Activation('relu')(row_cnn)
    row_cnn = Conv2D(56, (3, 28))(row_cnn)
    row_cnn = BatchNormalization()(row_cnn)
    row_cnn = Activation('relu')(row_cnn)
    row_cnn = Conv2D(56, (3, 19))(row_cnn)
    row_cnn = BatchNormalization()(row_cnn)
    row_cnn = Activation('relu')(row_cnn)
    row_cnn = Flatten()(row_cnn)
    row_cnn = Dense(4096, name = 'row_cnn_output')(row_cnn)
    
    col_cnn = Conv2D(56, (56, 3))(vanilla_cnn.get_layer('vanilla_cnn_output').output)
    col_cnn = BatchNormalization()(col_cnn)
    col_cnn = Activation('relu')(col_cnn)
    col_cnn = Conv2D(56, (28, 3))(col_cnn)
    col_cnn = BatchNormalization()(col_cnn)
    col_cnn = Activation('relu')(col_cnn)
    col_cnn = Conv2D(56, (19, 3))(col_cnn)
    col_cnn = BatchNormalization()(col_cnn)
    col_cnn = Activation('relu')(col_cnn)
    col_cnn = Flatten()(col_cnn)
    col_cnn = Dense(4096, name = 'col_cnn_output')(col_cnn)
    
    row_col_features = Add()([row_cnn, col_cnn])
    row_col_features = Activation('tanh')(row_col_features)
    row_col_features = Dense(512)(row_col_features)
    row_col_features = Activation('tanh')(row_col_features)
    row_col_features = Dense(256)(row_col_features)
    row_col_features = Activation('hard_sigmoid')(row_col_features)
    motion_features = Dense(30)(row_col_features)
    
    model = Model(inputs = vanilla_cnn.input, output = motion_features)
    
    return model