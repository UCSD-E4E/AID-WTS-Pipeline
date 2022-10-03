
#add comments
import os
import sys
import csv
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from Training_Models.tweetynet.chunking import extact_split, fill_no_class
import torch
from torch import nn
from torch.utils.data import DataLoader
from Training_Models.tweetynet.network import TweetyNet
import librosa
from librosa import display
from microfaune.audio import wav2spc, create_spec, load_wav
from glob import glob

from torch.utils.data import Dataset
from Training_Models.tweetynet.CustomAudioDataset import CustomAudioDataset

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import scipy
import IPython.display as ipd

from Training_Models.tweetynet.TweetyNetModel import TweetyNetModel  

from pydub import AudioSegment
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment

import os

from os import listdir
from os.path import isfile, join

import os

from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
import pydub


def load_dataset(data_path, use_dump=True, csv="labels.csv", kalediscope=False):
    mel_dump_file = os.path.join(data_path, "mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    
    elif kalediscope:
        dataset = compute_feature_kaledoscope_labels(data_path, csv)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    else:
        dataset = compute_feature(data_path, csv)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 431]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = [dataset["uids"][i] for i in inds]
    return X, Y, uids

def compute_feature(data_path, csv = "labels.csv"):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    labels_file = os.path.join(data_path, csv)
    print(labels_file)
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            labels = {}
            next(reader)  # pass fields names
            for name, _, y in reader:
                labels[name] = y
    else:
        print("Warning: no label file detected.")
        wav_files = glob(os.path.join(data_path, "wav/*.wav"))
        labels = {os.path.basename(f)[:-4]: None for f in wav_files}
    i = 1
    X = []
    Y = []
    uids = []
    for file_id, y in labels.items():
        print(f"{i:04d}/{len(labels)}: {file_id:20s}", end="\r")
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"), n_mels=n_mels)
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def compute_feature_df(data_path, df):
    
    i = 1
    X = []
    Y = []
    uids = []
    for file_id, y in labels.items():
        print(f"{i:04d}/{len(labels)}: {file_id:20s}", end="\r")
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"), n_mels=n_mels)
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test


def tweetynet_training(data, manual_df, species_of_interests):
    
    train = True
    fineTuning = False
    #needs at least 80 for mel spectrograms ## may be able to do a little less, but must be greater than 60
    n_mels=72 # The closest we can get tmeporally is 72 with an output of 432 : i think it depends on whats good
    #this number should be proportional to the length of the videos. 
    num_workers=2

    datasets_dir = "C:/Users/siloux/Desktop/E4E/passive-acoustic-biodiversity/TweetyNET/cosmos_data/Cosmos_data"
    this_is_new_data = True
    path_to_audio = "C:/Users/siloux/Desktop/E4E/passive-acoustic-biodiversity/TweetyNET/cosmos_data/cosmos_random_sample_processing"
    data_path = "C:/Users/siloux/Desktop/E4E/passive-acoustic-biodiversity/TweetyNET/cosmos_data/cosmos_random_sample_processing_split"
    data_path = "C:/Users/siloux/Desktop/E4E/passive-acoustic-biodiversity/TweetyNET/cosmos_data/cosmos_random_sample_processing_split"
    new_folder = data_path
    AudioSegment.converter = "C:/Users/siloux/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    pydub.AudioSegment.ffmpeg = "C:/Users/siloux/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    main_data_location = ".\cosmos_data\Cosmos_data"

    onlyfiles = [f for f in listdir(path_to_audio) if isfile(join(path_to_audio, f))]
    data = data[data["IN FILE"].isin(onlyfiles)]

    new_folder

    chunked_renamed_df = data.copy()
    count = 0
    for filename in np.unique(data["IN FILE"]):
        file_df = data[data["IN FILE"] == filename]
        if (len(filename.split(".")) != 2):
            break
        file, file_type = filename.split(".")
        for index,row in file_df.iterrows():
            t1 = row["CHUNK_ID"] * 1000
            # end time in milliseconds
            t2 = (row["CHUNK_ID"] + 3) * 1000
            new_name = file + "_" + str(int(row["CHUNK_ID"])) +".wav"
            new_path = new_folder + "/" + new_name
            
            try:
                if(not os.path.exists(new_path)):
                    newAudio = AudioSegment.from_mp3(path_to_audio + "/" + filename)
                    newAudio = newAudio[t1:t2]
                    newAudio = newAudio.set_channels(1)
                    newAudio.export(new_path, format="wav")

                chunked_renamed_df.at[index, "IN FILE"] = new_name
                chunked_renamed_df.at[index, "FOLDER"] = new_folder
            except:
                chunked_renamed_df.at[index, "IN FILE"] = "TO DELETE"
                chunked_renamed_df.at[index, "FOLDER"] = "TO DELETE"
            count += 1
            print(count, chunked_renamed_df.shape[0])
            
    chunked_renamed_df = chunked_renamed_df[chunked_renamed_df["IN FILE"] != "TO DELETE"]
    chunked_renamed_df["y"] = chunked_renamed_df["MANUAL ID"].apply(lambda x: species_of_interests.index(x))
    
    # You dont need the number of files in the folder, just iterate over them directly using:
    if (this_is_new_data):
        for index, row in chunked_renamed_df.iterrows():
            file = row["IN FILE"]
            folder = row["FOLDER"]
            #spliting the file into the name and the extension
            name, ext = os.path.splitext(folder + '/' + file)
            if ext == ".mp3":
                #os.remove(folder + '/' + file) 

                mp3_sound = AudioSegment.from_mp3(folder + '/' + file)
                mp3_sound.export("{0}.wav".format(name), format="wav")
    if(this_is_new_data):
        # You dont need the number of files in the folder, just iterate over them directly using:
        for index, row in chunked_renamed_df.iterrows():
            file = row["IN FILE"]
            folder = row["FOLDER"]
            #spliting the file into the name and the extension
            name, ext = os.path.splitext(file)
            if ext == ".mp3":
                print(folder)
                try:
                    os.remove(folder + "/" + file) 
                except Exception as e:
                    print(e)
    chunked_renamed_df["FOLDER"] = main_data_location
    chunked_renamed_df["IN FILE"] = chunked_renamed_df["IN FILE"].apply(lambda x: x.replace(".mp3", ".wav"))
    i = 1
    X = []
    Y = []
    uids = []
    for index, row in chunked_renamed_df.iterrows():
        try:
            print(f"{i:04d}/{chunked_renamed_df.shape[0]}", end="\r")
            print(os.path.join(data_path, row["IN FILE"]))
            spc = wav2spc(data_path + '/' + row["IN FILE"], n_mels=n_mels)
            X.append(spc)
            Y.append(row["y"])
            uids.append(row["IN FILE"])
            i += 1
        except Exception as e:
            print(e)
            continue
    
    dataset = {"uids": uids, "X": X, "Y": Y}
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 130]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = np.array([dataset["uids"][i] for i in inds])

    ind_train_val, ind_test = split_dataset(X, Y)
    ind_train, ind_val = split_dataset(X[ind_train_val, :, :, np.newaxis], Y[ind_train_val], test_size=0.1)
    X_train, X_test, X_val = X[ind_train, :, :, np.newaxis], X[ind_test, :, :, np.newaxis], X[ind_val, :, :, np.newaxis]
    Y_train, Y_test, Y_val = Y[ind_train], Y[ind_test], Y[ind_val]
    uids_train, uids_test, uids_val = uids[ind_train], uids[ind_test], uids[ind_val]
    #del X, Y

    print("Training set: ", Counter(Y_train))
    print("Test set: ", Counter(Y_test))
    print("Validation set: ", Counter(Y_val))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    tweetynet = TweetyNetModel(len(Counter(Y_train)), (1, n_mels, 431), device, binary=True, workers=num_workers)

    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)

    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)

    history, test_out, start_time, end_time = tweetynet.train_pipeline(train_dataset, val_dataset, test_dataset, 
                                                                    lr=.001, batch_size=128,epochs=10, save_me=True,
                                                                    fine_tuning=False, finetune_path=None)
    try:
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        predictions = pd.DataFrame()
        pred_one_hot = pd.DataFrame(columns=np.append("UID", test_dataset.unique_labels()))
        species = list(np.unique(chunked_renamed_df["MANUAL ID"]))
        tweetynet.model.eval()


        scores_df = pd.DataFrame()
        pred_df = pd.DataFrame()
        label_df = pd.DataFrame()

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                #ACTUALLY RUNNING MODEL
                #Run model over test data
                inputs, labels, uids = data
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                inputs, labels = inputs.to(tweetynet.device), labels.to(tweetynet.device)

                output = tweetynet.model(inputs, inputs.shape[0], labels.shape[0])
                
                #Maniplute data shape
                temp_uids = []
                if tweetynet.binary:
                    labels = labels.detach()
                    labels = torch.tensor([[x] * output.shape[-1] for x in labels]).to(tweetynet.device)
                    temp_uids = np.array([[x] * output.shape[-1] for x in uids])
                else:
                    for u in uids:
                        for j in range(output.shape[-1]):
                            temp_uids.append(str(j) + "_" + u)
                    temp_uids = np.array(temp_uids)
                
                
                #PREPRARE DATAFRAME FOR ONE HOT ENCODING VISULIZAITONS
                #Get the predictions for each
                d = {}
                p = {}
                l = {}
                label_d = {}
                scores = []
                
                prediction = torch.argmax(output, dim=1)
                d["uid"] =  temp_uids.flatten()
                p["uid"] =  temp_uids.flatten()
                for species_index in range(output.shape[1]):
                    #get scores from output
                    scores = []
                    preds = []
                    labels_ = []
                    test = output[:, species_index, :].numpy()
                    for file in test:
                        for slice_ in file:
                            scores.append(slice_)
                            preds.append(0)
                            labels_.append(0)
                            count += 1

                    d[species[species_index]] = scores
                    p[species[species_index]] = preds
                    l[species[species_index]] = labels_

                #Label predictions in one hot encoding
                new_scores = pd.DataFrame(d)
                new_pred = pd.DataFrame(p)
                new_labels = pd.DataFrame(l)
            
                counter = 0
                for index, row in new_pred.iterrows():
                    if (index % 144 == 0 and index != 0):
                        counter += 1
                    species_index = prediction[counter][index % 144]
                    row[species[species_index]] = 1
                    new_pred.loc[index] = row
                
                counter = 0
                for index, row in new_labels.iterrows():
                    if (index % 144 == 0 and index != 0):
                        counter += 1
                    species_index = labels[counter][index % 144]
                    row[species[species_index]] = 1
                    new_labels.loc[index] = row
                
                scores_df = scores_df.append(new_scores)
                pred_df = pred_df.append(new_pred)
                label_df = label_df.append(new_labels)
                print("next", i, len(test_loader))

        print('Finished Testing')
        scores_df, pred_df, label_df
    except:
        print("I STILL NEED TO FINSH THIS WORK")
    
    return tweetynet, history, test_out, start_time, end_time 

        