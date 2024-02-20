"""
    Train a multilayer perceptron classifier on the embedded training trials for each model, encoding, and difficulty
    Input: 
        Training trials containing embeddings for each pair of calls
    Output: 
        Multilayer Perceptron model for later evaluation
"""

import os
import numpy as np
import time
import yaml
import sys
from joblib import dump 
from sklearn.neural_network import MLPClassifier


###########################       Train classifier       ###########################

def concat_trial_embeds(train_trials_file, model):
    """ Concatenate embeddings for each trial in the evaluation set """
    with open(train_trials_file, 'rb') as f:
        train_trials = np.load(f, allow_pickle=True)
    X_train = []
    labels = []
    for trial in train_trials:
        trial_embed = np.concatenate((trial['call 1'][0], trial['call 2'][0]))
        X_train.append(trial_embed)
        labels.append(trial['label'])
    y_train = np.array(labels) 
    return X_train, y_train


def train_model_mlp(X_train, y_train, max_i, solvr, r_state, model_outfile): 
    """ Train MLP classifier and save to file """
    mlp = MLPClassifier(max_iter=max_i, solver=solvr, random_state=r_state) 
    mlp.fit(X_train, y_train) 
    dump(mlp, model_outfile)
    return 


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    work_dir = cfg['work_dir']
    encodings = cfg['encodings']
    difficulties = cfg['difficulties']
    r_state = cfg['r_state'] 
    models = cfg['models']
    clf = cfg['clf']
    solvr = cfg['solver']
    max_i = cfg['max_iter']

    ### Train a classifier for each model, encoding, and difficulty
    for model in models: 
        for encoding in encodings: 
            for difficulty in difficulties: 
                ### Infile
                model_dir = os.path.join(work_dir, f'{model}_model') 
                train_trials_file = os.path.join(model_dir,
                                            f'{encoding}_train_{difficulty}_trials.npy') 
                
                ### Outfile
                model_outfile = os.path.join(model_dir,
                                            f'{model}_{encoding}_{difficulty}_{clf}.joblib')
                
                X_train, y_train = concat_trial_embeds(train_trials_file, model)
                ### Train classifier
                tic = time.perf_counter() 
                train_model_mlp(X_train, y_train, max_i, solvr, r_state, model_outfile)
                toc = time.perf_counter()
                print(f"{model} {encoding} {difficulty} training time: {toc - tic:0.3f} seconds")
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)



