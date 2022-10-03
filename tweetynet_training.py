
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


    
    chunk  = 3

    data = data[["IN FILE", "FOLDER", "OFFSET", "DURATION", "MANUAL ID", "CONFIDENCE", "CLIP LENGTH"]]
    chunked_df = extact_split(data, chunk)
    chunked_df["CHUNK_ID"] = chunked_df["OFFSET"] // chunk
    chunked_df