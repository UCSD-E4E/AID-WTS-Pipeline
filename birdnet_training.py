import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Flatten, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from data_processing import readAudioData, splitSignal
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


from datetime import datetime
class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, output_size=10):
        self.X = X
        self.Y = Y
        self.n = len(Y)
        self.batch_size = batch_size
        self.output_size = output_size
        self.shuffle()
        
        
    def __len__(self):
        print("RUNNINGS", int(np.floor(self.n)/self.batch_size))
        return int(np.floor(self.n)/self.batch_size)
    
    def __getitem__(self, index):
        
        batch_inds = self.inds[self.batch_size*index:self.batch_size*(index+1)]
        tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])
        #print(index, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
        if (tf.reduce_max(tensor) == tf.reduce_min(tensor)):
            try:
                print("REDO")
                print(index, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
                batch_inds = self.inds[self.batch_size*(index+1):self.batch_size*(index+ 2)]
                tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])
                print(index+1, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
            except:
                batch_inds = self.inds[self.batch_size*(index-1):self.batch_size*(index)]
                tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])

        tensor = tf.math.divide(
           tf.math.subtract(
              tensor, 
              tf.reduce_min(tensor)
           ), 
           tf.math.subtract(
              tf.reduce_max(tensor), 
              tf.reduce_min(tensor)
           )
        )
        
        
        
        #print(batch_inds)
        self.counter += self.batch_size
        if self.counter >= self.n:
            self.shuffle()
            
        #TODO: FIX SO BATCH IS MORE THAN 1
        raw_labels = np.array([self.Y[batch_inds][0]])
        formatted_labels = np.zeros(self.output_size)
        formatted_labels[self.Y[batch_inds][0]] = 1
        fprmatted_labels = np.array([formatted_labels])
        #print(formatted_labels.shape)
        #print(np.array([formatted_labels]).shape)
        #print(self.X[batch_inds, ...][0].shape)

        
        #print(tf.convert_to_tensor(self.X[batch_inds, ...][0]), tf.convert_to_tensor(np.array([formatted_labels])))
        return tensor, tf.convert_to_tensor(np.array([formatted_labels]))
    
    def shuffle(self):
        self.inds = np.random.permutation(self.n)
        self.counter = 0


class ValidDataset(keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, output_size=10):
        self.X = X
        self.Y = Y
        #self.UIDs = UIDs
        self.n = len(Y)
        self.batch_size = batch_size
        self.output_size = output_size
        self.shuffle()
        
        
    def __len__(self):
        print("RUNNINGS", int(np.floor(self.n)/self.batch_size))
        return int(np.floor(self.n)/self.batch_size)
    
    def len_of_labels(self):
        return self.output_size
    
    def __getitem__(self, index):
        
        batch_inds = self.inds[self.batch_size*index:self.batch_size*(index+1)]
        tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])
        #print(index, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
        if (tf.reduce_max(tensor) == tf.reduce_min(tensor)):
            try:
                print("REDO")
                print(index, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
                batch_inds = self.inds[self.batch_size*(index+1):self.batch_size*(index+ 2)]
                tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])
                print(index+1, tf.reduce_max(tensor), tf.reduce_min(tensor), tf.reduce_max(tensor) == tf.reduce_min(tensor))
            except:
                batch_inds = self.inds[self.batch_size*(index-1):self.batch_size*(index)]
                tensor = tf.convert_to_tensor(self.X[batch_inds, ...][0])

        tensor = tf.math.divide(
           tf.math.subtract(
              tensor, 
              tf.reduce_min(tensor)
           ), 
           tf.math.subtract(
              tf.reduce_max(tensor), 
              tf.reduce_min(tensor)
           )
        )
        
        
        
        #print(batch_inds)
        self.counter += self.batch_size
        if self.counter >= self.n:
            self.shuffle()
            
        #TODO: FIX SO BATCH IS MORE THAN 1
        raw_labels = np.array([self.Y[batch_inds][0]])
        formatted_labels = np.zeros(self.output_size)
        formatted_labels[self.Y[batch_inds][0]] = 1
        fprmatted_labels = np.array([formatted_labels])
        #print(formatted_labels.shape)
        #print(np.array([formatted_labels]).shape)
        #print(self.X[batch_inds, ...][0].shape)

        
        #print(tf.convert_to_tensor(self.X[batch_inds, ...][0]), tf.convert_to_tensor(np.array([formatted_labels])))
        return tensor, tf.convert_to_tensor(np.array([formatted_labels]))
    
    
    def shuffle(self):
        self.inds = np.random.permutation(self.n)
        self.counter = 0



def create_dataset(df, species_of_interests):
    df["y"] = df["MANUAL ID"].apply(lambda x: list(species_of_interests).index(x))
    df

    X = []
    Y = []
    UID = []

    #assume df is chunked
    for file in np.unique(df["IN FILE"]):
        file_df = df[df["IN FILE"] == file]
        #print((file_df["FOLDER"] + file_df["IN FILE"]).iloc[0])
        try:
            chunks, clip_length = readAudioData((file_df["FOLDER"] + file_df["IN FILE"]).iloc[0], 0, sample_rate=48000)
        except:
            continue
        offset = 0
        for c in range(len(chunks)):
            offset = c * 3
            try:
                tmp = file_df[file_df["OFFSET"] == offset]

                if (tmp.empty):
                    continue
                chunk_df = tmp.sample(1)
                if (chunk_df.empty):
                    continue
                    #Wouldn't this miss no bird??????
                # Add to batch
                X.append(np.array(chunks[c]))
                Y.append(chunk_df["y"].iloc[0])
                UID.append(chunk_df.index[0])
            except Exception as e:
                print(e)
                print(file_df)
                continue
    dataset = {"X":X, "Y": Y, "uids": UID}
    inds = [i for i, x in enumerate(dataset["X"])]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = [dataset["uids"][i] for i in inds]
    return  X, Y, uids


def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train], X[ind_test]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def birdnet_training(chunked_df, manual_df, species_of_interests, test_no_bird=True, lr=0.001, epochs=4,batch_size=1, workers=6):
    #Format data for trainining
    train_dataset = create_dataset(chunked_df, species_of_interests)
    test_dataset = None
    if (test_no_bird):
        test_dataset = create_dataset(manual_df, list(np.append(species_of_interests, "no bird")))
    else:
        test_dataset = create_dataset(manual_df, species_of_interests)

    #spilt dataset
    X_train, Y_train = train_dataset[0][:, np.newaxis], train_dataset[1]  
    X_test,  Y_test = test_dataset[0][:, np.newaxis], test_dataset[1]  
    print("Training set: ", Counter(Y_train))
    print("Test set: ", Counter(Y_test))

    #Rebuild birdnet for our classes
    path_to_saved_model = ".Training_Models/birdnet_analyzer/checkpoints/V2.1/BirdNET_GLOBAL_2K_V2.1_Model"
    model = tf.keras.models.load_model(path_to_saved_model)
    model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=['accuracy'])
    x = model.layers[-2].output
    o = tf.keras.layers.Dense(len(species_of_interests), activation='sigmoid', name='its_new_lmao')(x)
    model_extended = Model(inputs=model.input, outputs=o)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    ##MAIN TRAINING CODE
    optimizer = keras.optimizers.Adam(lr=lr)
    model_extended.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', keras.metrics.FalseNegatives()])

    alpha = 0.5
    data_generator = DataGenerator(X_train, Y_train, batch_size, output_size=len(species_of_interests))
    
    micro_callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-5),
        #keras.callbacks.ModelCheckpoint('microfaune-' + date_str +'-{epoch:02d}.h5',
        #                          save_weights_only=False)
    ]
    
    #validation_data=(X_test, Y_test),
    
    history = model_extended.fit(data_generator, epochs=epochs,batch_size=batch_size,workers=workers,
                                  #validation_data=(X_test, Y_test), class_weight={0: alpha, 1: 1-alpha},callbacks=micro_callbacks,
                                   verbose=2)

    return model_extended, X_test,  Y_test

def birdnet_testing(model_extended, X_test,  Y_test, species_of_interests):
    validate_dataset = ValidDataset(X_test, Y_test, 1, output_size=len(species_of_interests))
    label_df = pd.DataFrame(columns=range(validate_dataset.len_of_labels()))
    scores_df = pd.DataFrame(columns=range(validate_dataset.len_of_labels()))
    preds_df = pd.DataFrame(columns=range(validate_dataset.len_of_labels()))
    for i in range(len(validate_dataset)):
        #print(i)
        predictions = model_extended.predict(
            validate_dataset.__getitem__(i)[0],
            batch_size=None,
            verbose='1',
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=6,
            use_multiprocessing=True
        )
        
        label = predictions.argmax()
        label_arr = np.zeros(validate_dataset.len_of_labels())
        label_arr[label] = 1
        
        preds_df = preds_df.append(pd.DataFrame(np.array([label_arr])))
        scores_df = scores_df.append(pd.DataFrame(predictions))
        label_df = label_df.append(pd.DataFrame(validate_dataset.__getitem__(i)[1].numpy()))
        
    preds_df.columns = np.append(species_of_interests, -1)
    scores_df.columns = np.append(species_of_interests, -1)
    label_df.columns = np.append(species_of_interests, -1)
    print(classification_report(label_df, preds_df))
    import matplotlib.pyplot as plt
    for species in species_of_interests:
        try:
            fpr, tpr, thresh = roc_curve(label_df[species],  scores_df[species])
            auc = roc_auc_score(label_df[species],  preds_df[species])
            plt.plot(fpr,tpr,label="AUC " + species + " "+str(auc))
        except:
            continue

    plt.title('Classwise ROC Curves')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
    return model_extended, label_df, scores_df, preds_df
