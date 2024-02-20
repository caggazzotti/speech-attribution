"""
    Iterate through each trial and get an SBERT embedding for each call in the trial
    SBERT model checkpoint: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2

    Input data: 
        transcript_trials = [{'label': 1, 'call 1': ['utterance 1', 'utterance 2', '...'], 
                            'call 2': ['utterance 1', 'utterance 2', '...']}, 
                            {...}, ...]
    Output data:
        embedded_trials = [{'label': 1, 'call 1': [embedding], 'call 2': [embedding]}, 
                            {...}, ...]
"""

import os
import numpy as np
import time
import yaml
import sys
import torch

# requirement: pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
  print("GPU information")
  print(torch.cuda.device_count())
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.get_device_name(0))
else:
  print("WARNING: No GPU detected")


###############################       SBERT embed trials       ###############################

def embed_trials(trials_file, model, embed_outfile): 
    """
    Embed each trial in the trials file using the SBERT model
    """
    embeddings = [] 
    with open(trials_file, 'rb') as f:
        trials = np.load(f, allow_pickle=True)
    for trial in trials:
        embedding1 = embed_utterances(trial['call 1'], model)
        embedding2 = embed_utterances(trial['call 2'], model)
        ### Ensure call has an embedding (since remove first 5 utterances, call could be too short)
        if not np.any(np.isnan(embedding1)) and not np.any(np.isnan(embedding2)):
            embedded_trial = {'label': trial['label'],
                                'call 1': embedding1,
                                'call 2': embedding2}
            embeddings.append(embedded_trial)
    print('Num of trials embedded: ', len(embeddings)) 
    output_to_file(embeddings, embed_outfile)
    return 


def embed_utterances(speaker_utterances, model):
    """ 
    Get pointwise average of SBERT embeddings for each speaker's utterances in a call
    """
    utterance_embeddings = []
    for utterance in speaker_utterances:
        embedding = model.encode([utterance.strip()]) 
        utterance_embeddings.append(embedding) 
    avg_embedding = np.mean(utterance_embeddings, axis=0) # average along column
    return avg_embedding


###########################       Save and output       ###########################

def output_to_file(data, data_outfile): 
    np.save(data_outfile, data, allow_pickle=True)
    return


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    work_dir = cfg['work_dir']
    datasets = cfg['datasets']
    encodings = cfg['encodings']
    difficulties = cfg['difficulties']

    trials_dir = os.path.join(work_dir, 'trials_data')
    if not os.path.exists('SBERT_model'):
        os.makedirs('SBERT_model')
    
    ### Load the model
    SBERT_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    ### Embed each set of trials
    for encoding in encodings: # 'bbn', 'ldc'
        for dataset in datasets: # 'train', 'val', 'test' 
            for difficulty in difficulties: # 'base', 'hard', 'harder'
                ### Infile
                trials_file = os.path.join(trials_dir, 
                                        f'{encoding}_{dataset}_{difficulty}_trials.npy')

                ### Outfile 
                embed_outfile = os.path.join(work_dir, 
                                        f'SBERT_model/{encoding}_{dataset}_{difficulty}_trials.npy')
                
                ### Embed trials
                tic = time.perf_counter()
                print(f"SBERT embedding {encoding} {dataset} {difficulty} trials...")
                embed_trials(trials_file, SBERT_model, embed_outfile)
                toc = time.perf_counter()
                print(f"Embedded trials in {(toc - tic)/60:0.3f} minutes")
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)

