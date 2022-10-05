import pandas as pd
import numpy as np
import pandas as pd
from opensoundscape.torch.models import cnn
from opensoundscape.preprocess.preprocessors import BasePreprocessor, CnnPreprocessor
import torch
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from opensoundscape.annotations import categorical_to_one_hot
import matplotlib.pyplot as plt
from opensoundscape.torch.models.cnn import load_model
from opensoundscape.audio import Audio
import os

def prep_data(df, ef, name = "autoamted_cosmos_tweety_to_file", THIS_IS_NEW_DATA = True):
    #COSMOS_BirdNET-Lite_Labels_100.csv
    
    test_name = "./" +name + "_TESTING.csv" #"./COSMOS_BirdNET-Lite_Labels_100_TESTING.csv"
    train_name ="./" +name + "_TRAINING.csv" #"./COSMOS_BirdNET-Lite_Labels_100_TRAINING.csv"
    
    
    print(df["IN FILE"])

    ef['IN FILE'] = ef['IN FILE'].apply(lambda x : x.replace("_", " "))
    gtClips = np.array(ef['IN FILE'].unique())
    j = 0
    ff = ef
    
    #pd.DataFrame(columns=df.columns)
    #for i, row in ef.iterrows():
    #    if(row['IN FILE'] in gtClips):
    #        # print(row['IN FILE'])
    ##        ff.loc[j] = row
    #        df.drop(i, inplace=True)
    #        j += 1
    #print(df.shape)

    try:
        df = df[df['CONFIDENCE'] >= 0.8]
    except:
        pass

    df['IN FILE'] = ['./Cosmos_data/Training/' + f for f in df['IN FILE']]
    ff['IN FILE CURRENT'] = ['./Cosmos_data/Training_Xeno_Canto_2022/' + f for f in ff['IN FILE']]
    ff['IN FILE'] = ['./Cosmos_data/Testing_Xeno_Canto_2022/' + f for f in ff['IN FILE']]


    ef = ff.groupby('IN FILE', group_keys=False).apply(lambda ff: ff.sample(1))
    #ef['IN FILE'] = ['./XenoCanto_Data/Testing_Xento_Canto_2022/' + f for f in ef['IN FILE']]
    ef

    import shutil

    def Move_Files(filename):
        file = filename.split("/")
        og_file_path = "/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/" + file[3]
        new_file_path = "/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/" + "/".join(file[2:])
        if(THIS_IS_NEW_DATA):
            shutil.copyfile(og_file_path, new_file_path)
        return new_file_path
    #Move_Files('./Cosmos_data/Testing_Xeno_Canto_2022/XC100864 - Rufous-collared Sparrow - Zonotrichia capensis costaricensis.mp3')
    #shutil.copyfile('Cosmos_data/XC100864 - Rufous-collared Sparrow - Zonotrichia capensis costaricensis.mp3', './Cosmos_data/Testing_Xeno_Canto_2022/XC100864 - Rufous-collared Sparrow - Zonotrichia capensis costaricensis.mp3')
    #shutil.copyfile("/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/XC100864 - Rufous-collared Sparrow - Zonotrichia capensis costaricensis.mp3", '/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/Testing_Xeno_Canto_2022/XC100864 - Rufous-collared Sparrow - Zonotrichia capensis costaricensis.mp3')
    ef["IN FILE"] = ef["IN FILE"].apply(Move_Files)

    clip_duration = 3
    clip_overlap = 0
    final_clip = None
    clip_dir = './temp_clips/Cosmos_Data/Testing_Xeno_Canto_2022/'
    # classes = labels
    min_label_overlap = 0.1

    ef_split_save = pd.DataFrame(columns=['file', 'start_time', 'end_time', 'SAMPLING RATE', 'MANUAL ID'])

    cnt = 0
    ef = ef.reset_index()
    print(ef)
    for i, row in ef.iterrows():
        try:
            audio = Audio.from_file(row['IN FILE'])
        except Exception:
            print(row['IN FILE'] + ' not found')
            continue
        file_name = row['IN FILE'].split('/')
        # print(file_name)
        clip_df = audio.split_and_save(
            clip_dir,
            prefix=file_name[-1],
            clip_duration=clip_duration,
            clip_overlap=clip_overlap,
            final_clip='remainder',
            dry_run=(os.path.exists(clip_dir + file_name[-1]))
        )
        clip_df['SAMPLING RATE'] = [44100] * clip_df.shape[0]
        clip_df['MANUAL ID'] = [row['MANUAL ID']] * clip_df.shape[0]
        clip_df.reset_index(inplace=True)
        ef_split_save = ef_split_save.append(clip_df, ignore_index=True)
        print(i, "out of", ef.shape[0])

    print(ef_split_save)
    ef_split_save.to_csv(test_name)

    import shutil

    def Move_Files(filename):
        file = filename.split("/")
        og_file_path = "/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/" + file[3]
        new_file_path = "/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/" + "/".join(file[2:])
        if(not os.path.exists(new_file_path)):
            shutil.copyfile(og_file_path, new_file_path)
        return new_file_path

    df['IN FILE'] = [f.split('/')[-1] for f in df['IN FILE']]
    df['IN FILE'] = ['./Cosmos_Data/Training_Xeno_Canto_2022/' + f for f in df['IN FILE']]
    df["IN FILE"] = df["IN FILE"].apply(Move_Files)
    df

    def fix_folder_path(filepath):
        folder = "/home/shperry/passive-acoustic-biodiversity/OpenSoundScape/MultiClass Classifier/Cosmos_Data/Training_Xeno_Canto_2022/"
        filename = filepath.split("/")[-1]
        return folder + filename
    df["IN FILE"] = df["IN FILE"].apply(fix_folder_path)
    df

    if (False):
        from opensoundscape.audio import Audio
        clip_duration = 3
        clip_overlap = 0
        final_clip = None
        clip_dir = './temp_clips/Cosmos_Data/Training_Xeno_Canto_2022'
        # classes = labels
        min_label_overlap = 0.1

        df_split_save = pd.DataFrame(columns=['file', 'start_time', 'end_time', 'SAMPLING RATE', 'MANUAL ID', "CONFIDENCE"])

        cnt = 0
        df = df.reset_index()
        for i, row in df.iterrows():
            try:
                audio = Audio.from_file(row['IN FILE'])
            except Exception:
                print(row['IN FILE'] + ' not found')
                continue
            file_name = row['IN FILE'].split('/')
            # print(file_name)
            clip_df = pd.DataFrame()
            offset = row["OFFSET"]
            duration = row["DURATION"]
            new_file = clip_dir + file_name[-1] + str(offset) + "s_" + str(offset+duration)+"s.wav"
            if (not os.path.exists(new_file)):
                clip_df = audio.split_and_save(
                    clip_dir,
                    prefix=file_name[-1],
                    clip_duration=clip_duration,
                    clip_overlap=clip_overlap,
                    final_clip='remainder',
                    dry_run=(os.path.exists(clip_dir + file_name[-1]))
                )
            else:
                tmp_df = pd.DataFrame()
                offset = row["OFFSET"]
                duration = row["DURATION"]
                tmp_df['file'] = new_file
                tmp_df['start_time'] = offset
                tmp_df['end_time'] = offset + duration
                if (clip_df.empty):
                    clip_df = tmp_df
                else:
                    clip_df = clip_df.append(tmp_df)

            clip_df['SAMPLING RATE'] = [44100] * clip_df.shape[0]
            clip_df['MANUAL ID'] = [row['MANUAL ID']] * clip_df.shape[0]
            clip_df.reset_index(inplace=True)
            df_split_save = df_split_save.append(clip_df, ignore_index=True)
            print(i, "out of", df.shape[0])

        print(df_split_save)

    !pip install pydub
    from opensoundscape.audio import Audio
    from pydub import AudioSegment
    #AudioSegment.converter = "C:/Users/Siloux/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    import os


    clip_duration = 3
    clip_overlap = 0
    final_clip = None
    clip_dir = './temp_clips/Cosmos_Data/Training_Xeno_Canto_2022'
    # classes = labels
    min_label_overlap = 0.1

    df_split_save = pd.DataFrame(columns=['file', 'start_time', 'end_time', 'SAMPLING RATE', 'MANUAL ID', "CONFIDENCE"])

    def convert_df_to_ops_df(row):
        offset = row["OFFSET"]
        duration = row["DURATION"]
        in_file_curr = row['IN FILE']
        file_name = in_file_curr.split('/')
        new_file = clip_dir + file_name[-1] + "_" + str(offset) + "s_" + str(offset+duration)+"s.wav"
        row['IN FILE'] = new_file
        
        if(not os.path.exists(new_file)):
            t1 = row["OFFSET"] * 1000
            # end time in milliseconds
            t2 = (row["OFFSET"] + 3) * 1000
            try:
                newAudio = AudioSegment.from_mp3(in_file_curr)
                newAudio = newAudio[t1:t2]
                new_file_object = newAudio.export(new_file, format="mp3")
                new_file_object.close()
            except Exception as e:
                print("failed on", in_file_curr)
                row['IN FILE'] = "TO DELETE"
            #newAudio.close()
        
        return row
        

    df_refractor = df.apply(convert_df_to_ops_df, axis=1)
    df_refractor = df_refractor[df_refractor["IN FILE"] != "TO DELETE"]
    df_refractor

    df_split_save = df_refractor[["IN FILE", "OFFSET", "DURATION", "SAMPLE RATE", "MANUAL ID"]]
    df_split_save["end_time"] = df_split_save["OFFSET"] + df_split_save["DURATION"]
    df_split_save = df_split_save.rename(columns={"IN FILE": "file", "OFFSET": "start_time"})
    df_split_save = df_split_save[['file', 'start_time', 'end_time', 'SAMPLE RATE', 'MANUAL ID']]

    df_split_save.to_csv(train_name)
    df_split_save

    df["merge file"] = df["IN FILE"].apply(lambda x: x.split("/")[-1])
    df_split_save['CONFIDENCE'] = df_split_save['CONFIDENCE'].fillna(-1)

    df_split_save.drop_duplicates()

    import os
    from os import listdir
    from os.path import isfile, join
    test_path = "./temp_clips/Cosmos_Data/Testing_Xeno_Canto_2022"
    test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    train_path = "./temp_clips/Cosmos_Data/Training_Xeno_Canto_2022"
    train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    overlapping_files = []
    for item in train_files:
        if (item in test_files):
            path = train_path + "/" + item
            os.remove(path)
            overlapping_files.append(path)

    df_split_save2 = df_split_save[~df_split_save["file"].isin(overlapping_files)]
    df_split_save2.to_csv(train_name)       
    print(len(overlapping_files))
    df_split_save2

    test_path2 = "./temp_clips/Cosmos_Data/Testing_Xeno_Canto_2022"
    test_files2 = [f for f in listdir(test_path2) if isfile(join(test_path2, f))]
    train_path2 = "./temp_clips/Cosmos_Data/Training_Xeno_Canto_2022"
    train_files2 = [f for f in listdir(train_path2) if isfile(join(train_path2, f))]
    overlapping_files2 = []
    for item in train_files2:
        if (item in test_files2):
            overlapping_files2.append(item)
    overlapping_files2

    df_split_save2 = df_split_save[~df_split_save["file"].isin(overlapping_files)]
    df_split_save2.to_csv(train_name)       
    print(len(overlapping_files))

    df_split_save2.to_csv(train_name) 

def run_training():
    name = "autoamted_cosmos_tweety_to_file"
    train_val_df = pd.read_csv(name + '_TRAINING.csv')
    test_df = pd.read_csv(name + '_TESTING.csv')
    model_save = './model/' + name  + '_no_none_class/'
    train_val_df["filename"] = train_val_df["file"].apply(lambda x: x.split("/")[-1])
    test_df["filename"] = test_df["file"].apply(lambda x: x.split("/")[-1])
    train_val_df = train_val_df[~train_val_df["filename"].isin(test_df["filename"])]
    test_df.merge(train_val_df, left_on="filename", right_on="filename")
    classes = np.unique(test_df['MANUAL ID'])#['Antwren', 'Antshrike', 'Toucan', 'Vireo', 'Kingbird', 'Tody-Tyrant', 'None']
    train_val_df = train_val_df[train_val_df["MANUAL ID"].isin(classes)]
    test_df = test_df[test_df["MANUAL ID"].isin(classes)]
    print('Classwise Counts for train/val data:')
    print(train_val_df['MANUAL ID'].value_counts())
    print('\nClasswise  Counts for test data:')
    print(test_df['MANUAL ID'].value_counts())

    # Train/val
    one_hot_labels, train_classes = categorical_to_one_hot(train_val_df[['MANUAL ID']].values)
    train_val_df = pd.DataFrame(index=train_val_df['file'],data=one_hot_labels,columns=train_classes)

    # Test
    one_hot_labels, test_classes = categorical_to_one_hot(test_df[['MANUAL ID']].values)
    test_df = pd.DataFrame(index=test_df['file'],data=one_hot_labels,columns=test_classes)
    train_df, valid_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    print("Number of training examples : ", train_df.shape[0])
    print("Number of validation examples : ", valid_df.shape[0])
    print("Number of test examples : ", test_df.shape[0])

    train_dataset = CnnPreprocessor(df=train_df)
    train_dataset.augmentation_on()
    train_dataset.actions.load_audio.set(sample_rate=44100)
    valid_dataset = CnnPreprocessor(df=valid_df)
    valid_dataset.augmentation_on()
    valid_dataset.actions.load_audio.set(sample_rate=44100)

    model = cnn.Resnet18Multiclass(train_classes)

    model.optimizer_params = {
        "feature": {  # optimizer parameters for feature extraction layers
            # "params": self.network.feature.parameters(),
            "lr": 0.0001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
        "classifier": {  # optimizer parameters for classification layers
            # "params": self.network.classifier.parameters(),
            "lr": 0.0001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
    }

    model.sampler = 'imbalanced'
    model.train(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        save_path=model_save,
        batch_size=32,
        save_interval=100,
        num_workers=6,
        epochs=3
    )

    plt.plot(model.loss_hist.keys(),model.loss_hist.values())
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Plot of Loss vs Epochs')
    plt.show()

    return model

def testing(model, test_df):
    prediction_dataset = model.train_dataset.sample(n=0)
    prediction_dataset.augmentation_off()
    prediction_dataset.df = test_df

    prediction_dataset.df = prediction_dataset.df[list(model.classes)]
    valid_scores_df, valid_preds_df, valid_labels_df = model.predict(prediction_dataset,
                                                                 binary_preds='single_target',
                                                                 batch_size=16,
                                                                 num_workers=6,
                                                                 activation_layer='softmax')
    
    print(classification_report(valid_labels_df, valid_scores_df.apply(round)))

    for species in model.classes:
        fpr, tpr, thresh = roc_curve(valid_labels_df[species],  valid_scores_df[species])
        auc = roc_auc_score(valid_labels_df[species],  valid_preds_df[species])
        plt.plot(fpr,tpr,label="AUC " + species + " "+str(round(auc)))

    plt.title('Classwise ROC Curves')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.show()
