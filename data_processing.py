import librosa
import numpy as np

def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

def readAudioData(path, overlap, sample_rate=48000):

    #print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    try:
        sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')
        clip_length = librosa.get_duration(y=sig, sr=rate)
    except:
        return 0
    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks, clip_length

from birdnet_training import birdnet_testing, birdnet_training
from tweetynet_training import tweetynet_training

def general_training_testing(chunked_df, manual_df, species_of_interests, 
                        model="birdnet",
                        test_no_bird=True, 
                        lr=0.001, 
                        epochs=4,
                        batch_size=1, 
                        workers=6):
    if (model == "birdnet"):
        model_extended, X_test,  Y_test =  birdnet_training(chunked_df, manual_df, species_of_interests, 
                            test_no_bird=test_no_bird, 
                            lr=lr, 
                            epochs=epochs,
                            batch_size=batch_size, 
                            workers=workers)
        model_extended, label_df, scores_df, preds_df = birdnet_testing(model_extended, X_test,  Y_test, species_of_interests)
        return model_extended, label_df, scores_df, preds_df
    elif(model == "tweetynet"):
        return tweetynet_training(chunked_df, manual_df, species_of_interests)
        pass
    elif(model == "opensoundscape"):
        #TODO
        pass
    else:
        print("please set model to birdnet, tweetynet, or opensoundscape")
