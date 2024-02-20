"""
    Evaluate out-of-the-box and fine-tuned models using bootstrapped AUC and its 95% CI and standard error. Save results to txt file.

    Input data: 
        embedded_trials = [{'label': 1, 'call 1': [embedding], 'call 2': [embedding]},
                            {...}, ...]
    Output:
        txt file with bootstrapped metrics for each model's trials
"""

import os
import yaml
import sys
import numpy as np
import time
from joblib import load
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


###########################     Out-of-the-box evaluation      ###########################

def eval_cos_sim(eval_trials_file):
    """ Calculate cosine similarity for each trial in the evaluation set """
    with open(eval_trials_file, 'rb') as f:
        eval_trials = np.load(f, allow_pickle=True)
    sims = []
    labels = []
    for trial in eval_trials:
        sim = cosine_similarity(trial['call 1'].reshape(1, -1), trial['call 2'].reshape(1, -1))
        sims.append(float(sim))
        labels.append(trial['label'])
    y_true = np.array(labels)
    y_pred = np.array(sims)
    return y_true, y_pred


def bootstrap_auc_o(y_true, y_pred, nsamples):
    """ Bootstrap evaluation of AUC for out-of-the-box models """
    auc_values = []
    for b in range(nsamples): # with replacement so there are repeated indices
        idx = np.random.randint(y_true.shape[0], size=y_true.shape[0]) 
        roc_auc = roc_auc_score(y_true[idx].ravel(), y_pred[idx].ravel())
        auc_values.append(roc_auc)
    ci = np.percentile(auc_values, (2.5, 97.5)) # 95% CI
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)
    std_error = auc_std / np.sqrt(y_true.shape[0]) #sqrt(n) where n is # of samples
    results = [auc_mean, auc_std, std_error, ci]
    return results


###########################     Fine-tuned evaluation      ###########################

def concat_trial_embeds(eval_trials_file):
    """ Concatenate embeddings for each pair of calls in the evaluation set """
    with open(eval_trials_file, 'rb') as f:
        eval_trials = np.load(f, allow_pickle=True)
    embeds = []
    labels = []
    for trial in eval_trials:
        eval_embed = np.concatenate((trial['call 1'][0], trial['call 2'][0]))
        embeds.append(eval_embed)
        labels.append(trial['label'])
    X_eval = np.array(embeds) 
    y_eval = np.array(labels)   
    return X_eval, y_eval


def bootstrap_auc_ft(clf_file, X_eval, y_eval, nsamples):
    """ Bootstrap evaluation of AUC for fine-tuned models """
    clf = load(clf_file)
    y_pred = clf.predict_proba(X_eval)[:, 1]
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_eval.shape[0], size=X_eval.shape[0])
        roc_auc = roc_auc_score(y_eval[idx].ravel(), y_pred[idx].ravel())
        auc_values.append(roc_auc)
    ci = np.percentile(auc_values, (2.5, 97.5)) # 95% CI
    auc_mean = np.mean(auc_values)
    auc_std = np.std(auc_values)
    std_error = auc_std / np.sqrt(X_eval.shape[0]) #sqrt(n) where n is # of samples
    results = [auc_mean, auc_std, std_error, ci]
    return results


###########################       Save and output       ###########################

def output_to_file(data, data_outfile): 
    np.save(data_outfile, data, allow_pickle=True)
    return


def output_txt(bootstrap_results, model, model_version, eval_type, clf, solvr, max_i, 
               num_resamples, results_outfile): 
    """ Output bootstrap evaluation results to txt file """
    with open(results_outfile, 'w') as o_f:
        o_f.write('%s_%s Bootstrap Evaluation Results (%s set)\n' % (model, model_version, 
                                                                    eval_type))
        o_f.write('Classifier: %s\n' % clf)
        o_f.write('Solver: %s\n' % solvr)
        o_f.write('Num max iterations: %i\n' % max_i)
        o_f.write('---------------------------\n\n')
        for group in bootstrap_results:
            o_f.write('Encoding: %s\n' % group['encoding'])
            o_f.write('Difficulty: %s\n' % group['difficulty'])
            o_f.write('Num re-samples: %i\n' % num_resamples)
            o_f.write('Mean AUC: %f\n' % group['stats'][0])
            o_f.write('AUC standard deviation: %f\n' % group['stats'][1])
            o_f.write('AUC standard error: %f\n' % group['stats'][2])
            o_f.write('AUC .95 CI: %s\n' % str(group['stats'][3]))
            o_f.write('---------------------------\n\n')
    return


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    work_dir = cfg['work_dir']
    encodings = cfg['encodings']
    difficulties = cfg['difficulties']
    models = cfg['models']
    eval_type = cfg['eval_type']
    model_versions = cfg['model_versions']
    clf = cfg['clf']
    solvr = cfg['solver']
    max_i = cfg['max_iter']
    num_resamples = cfg['num_resamples']

    ### Evaluate each model's trials
    for model in models: # 'SBERT', 'CISR', 'LUAR', 'TFIDF'
        for model_version in model_versions: #'o', 'ft'
            bootstrap_results = []
            for encoding in encodings: # 'bbn', 'ldc'
                for difficulty in difficulties: # 'base', 'hard', 'harder'

                    ### Infiles
                    model_dir = os.path.join(work_dir, f'{model}_model') 
                    eval_trials_file = os.path.join(model_dir, 
                                            f'{encoding}_{eval_type}_{difficulty}_trials.npy') 
                    
                    ### Run bootstrapped evaluation
                    tic = time.perf_counter() 
                    if model_version == 'o': # out-of-the-box models
                        y_true, y_pred = eval_cos_sim(eval_trials_file) 
                        results = bootstrap_auc_o(y_true, y_pred, num_resamples)
                        bootstrap_results.append({'model': model, 'encoding': encoding, 
                                                  'difficulty': difficulty, 'stats': results})
                    elif model_version == 'ft': # fine-tuned models
                        clf_file = os.path.join(model_dir,
                                                f'{model}_{encoding}_{difficulty}_mlp.joblib')
                        X_eval, y_eval = concat_trial_embeds(eval_trials_file)
                        results = bootstrap_auc_ft(clf_file, X_eval, y_eval, num_resamples)
                        bootstrap_results.append({'model': model, 'encoding': encoding, 
                                                  'difficulty': difficulty, 'stats': results})
                    toc = time.perf_counter()
                    print(f"{model} {encoding} {difficulty} {model_version} eval time: {toc - tic:0.3f}s")

            ### Output results to txt file
            results_outfile = os.path.join(model_dir, 
                                    f'{model}_{eval_type}_bootstrap_{model_version}_results.txt')
            output_txt(bootstrap_results, model, model_version, eval_type, clf, solvr, max_i,
                        num_resamples, results_outfile)
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)

