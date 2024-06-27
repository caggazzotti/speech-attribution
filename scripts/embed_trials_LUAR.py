"""
    Iterate through each trial and get a LUAR (MUD) embedding for each call in the trial
    LUAR model checkpoint: https://huggingface.co/rrivera1849/LUAR-MUD

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

from transformers import AutoModel, AutoTokenizer

if torch.cuda.is_available():
    print("GPU information")
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
else:
    print("WARNING: No GPU detected")


###############################       LUAR embed trials       ###############################

def embed_trials(trials_file, tokenizer, model, embed_outfile): 
    """
    Embed each trial in the trials file using the LUAR model
    """
    embeddings = [] 
    with open(trials_file, 'rb') as f:
        trials = np.load(f, allow_pickle=True)
    for trial in trials:
        embedding1 = embed_utterances(trial['call 1'], tokenizer, model)
        embedding2 = embed_utterances(trial['call 2'], tokenizer, model)
        ### Ensure call has an embedding (since remove first 5 utterances, call could be too short)
        if len(embedding1) != 0 and len(embedding2) != 0:
            embedded_trial = {'label': trial['label'],
                                'call 1': embedding1,
                                'call 2': embedding2}
            embeddings.append(embedded_trial)
    print('Num of trials embedded: ', len(embeddings)) 
    output_to_file(embeddings, embed_outfile)
    return 


def embed_utterances(speaker_utterances, tokenizer, model):
    """ 
    Tokenize and get LUAR embeddings for each speaker's utterances in a call
    """
    batch_size = 128
    tokenized_text = tokenizer(
        speaker_utterances, 
        max_length=512, 
        padding="max_length", 
        truncation=True,
        return_tensors="pt"
    )
    episode_length = len(speaker_utterances) # speaker_utterances is one episode 
    ### Inputs size: (batch_size, episode_length, max_token_length)
    tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(batch_size, 
                                                                      episode_length, -1)
    tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(batch_size,
                                                                                episode_length, -1)
    with torch.inference_mode():    
        embedding = model(tokenized_text["input_ids"], tokenized_text["attention_mask"])
    return embedding


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
    if not os.path.exists('LUAR_model'):
        os.makedirs('LUAR_model')
    
    ### Load the tokenizer and model
    LUAR_tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")
    LUAR_model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    LUAR_model.eval()

    ### Embed each set of trials
    for encoding in encodings: # 'bbn', 'ldc'
        for dataset in datasets: # 'train', 'val', 'test' 
            for difficulty in difficulties: # 'base', 'hard', 'harder'
                ### Infile
                trials_file = os.path.join(trials_dir, 
                                        f'{encoding}_{dataset}_{difficulty}_trials.npy')

                ### Outfile 
                embed_outfile = os.path.join(work_dir, 
                                        f'LUAR_model/{encoding}_{dataset}_{difficulty}_trials.npy')
                
                ### Embed trials
                tic = time.perf_counter()
                print(f"LUAR embedding {encoding} {dataset} {difficulty} trials...")
                embed_trials(trials_file, LUAR_tokenizer, LUAR_model, embed_outfile)
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

