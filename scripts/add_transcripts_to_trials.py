"""
    This script retrieves the transcript for each call in each set of positive and negative trials. Not all BBN transcripts exist, so the number of trials reduces after adding BBN transcripts to the trials. Adding BBN transcripts must be done first so the LDC trials match the BBN trials.

    Inputs: 
        Dataset files containing positive and negative trial info 
    Outputs: 
        Dataset files containing positive and negative trials with corresponding BBN/LDC transcripts
        Dataset files containing positive and negative trial info - FINAL 
"""

import os
import numpy as np
import json
import time
import yaml
import sys
import re


##########################       Retrieve transcripts for trials       ##########################

def get_pos_transcripts(encoding, trial_type, pos_trials_info_file, trunc_style, trunc_size, 
                        dir1, dir2, transcripts_outfile):
    """
    Retrieve transcripts for each call in the positive trials and output to json file. Returns a final list of trials info (to use in retrieving LDC transcripts) after checking the BBN transcript exists for each call.

    Input data:
        pos_trials_info = [{'PIN': '#', 'call 1': [gender, call_ID, channel, topic], 
                        'call 2': [gender, call_ID, channel, topic]}, 
                        {...}, ...]
    Output data:
        pos_transcripts = [{'label': 1, 'call 1': ['utterance 1', 'utterance 2', '...'], 
                            'call 2': ['utterance 1', 'utterance 2', '...']}, 
                            {...}, ...]
    """
    with open(pos_trials_info_file, 'r') as f:
        pos_trials_info = json.loads(f.read())
    pos_transcripts = []
    pos_trials_final = []
    for trial in pos_trials_info:
        transcripts = []
        call_ID1 = trial['call 1'][1]
        channel1 = trial['call 1'][2]
        call_ID2 = trial['call 2'][1]
        channel2 = trial['call 2'][2]
        calls = [[call_ID1, channel1],[call_ID2, channel2]]
        for call in calls:
            if encoding == 'bbn':
                recreated_fname = 'fe_03_' + call[0] + '.txo'
                if int(call[0]) < 5851: # Fisher pt. 1: call IDs <= 05850
                    directory = dir1
                else: #Fisher pt. 2: call IDs > 05850
                    directory = dir2
                for folder in os.listdir(directory): 
                    folder_path = os.path.join(directory, folder) 
                    full_folder_path = os.path.join(folder_path, 'originals/')  
                    for filename in os.listdir(full_folder_path): 
                        if filename == recreated_fname:
                            fname = os.path.join(full_folder_path, filename)
                            if trunc_style == 'none':
                                speaker_lines = get_speaker_lines(fname, encoding, call[1])
                            elif trunc_style == 'beginning':
                                speaker_lines = get_speaker_lines_trunc_beg(fname, encoding, 
                                                                            call[1], trunc_size)
                            if speaker_lines: # make sure speaker has (enough) lines
                                transcripts.append(speaker_lines)
                            break
                    else:
                        continue
                    break
            elif encoding == 'ldc':
                recreated_fname = 'fe_03_' + call[0] + '.txt'
                if int(call[0]) < 5851: # Fisher pt. 1: call IDs <= 05850
                    directory = dir1
                else: #Fisher pt. 2: call IDs > 05850
                    directory = dir2
                for folder in os.listdir(directory): 
                    folder_path = os.path.join(directory, folder) 
                    for filename in os.listdir(folder_path): 
                        if filename == recreated_fname:
                            fname = os.path.join(folder_path, filename)
                            if trunc_style == 'none':
                                speaker_lines = get_speaker_lines(fname, 'ldc', call[1])
                            elif trunc_style == 'beginning':
                                speaker_lines = get_speaker_lines_trunc_beg(fname, 'ldc', 
                                                                            call[1], trunc_size)
                            if speaker_lines: # make sure speaker has (enough) lines
                                transcripts.append(speaker_lines)
                            break
                    else:
                        continue
                    break
        if len(transcripts) == 2: # make sure BBN transcript exists for both calls
            pos_transcript = {'label': 1,
                            'call 1': transcripts[0],
                            'call 2': transcripts[1]}
            pos_transcripts.append(pos_transcript)
            pos_trials_final.append(trial)
    output_to_file(pos_transcripts, transcripts_outfile) # trials with transcripts
    print(f'Number {trial_type} trials: ', len(pos_transcripts))
    return pos_trials_final, len(pos_transcripts)


def get_neg_transcripts(encoding, trial_type, neg_trials_info_file, trunc_style, trunc_size, 
                        dir1, dir2, transcripts_outfile):
    """
    Retrieve transcripts for each call in the negative trials and output to json file. Returns a final list of trials info (to use in retrieving LDC transcripts) after checking the BBN transcript exists for each call.

    Input data:
        neg_trials_info = [ [[pin, gender, call_ID, channel, topic],
                                [pin, gender, call_ID, channel, topic]], 
                               [ [],[] ], ...]
    Output data:
        neg_transcripts = [{'label': 0, 'call 1': ['utterance 1', 'utterance 2', '...'], 
                            'call 2': ['utterance 1', 'utterance 2', '...']}, 
                            {...}, ...]
    """
    with open(neg_trials_info_file, 'r') as f:
        neg_trials_info = json.loads(f.read())
    neg_transcripts = []
    neg_trials_final = []
    for trial in neg_trials_info:
        transcripts = []
        call_ID1 = trial[0][2]
        channel1 = trial[0][3]
        call_ID2 = trial[1][2]
        channel2 = trial[1][3]
        calls = [[call_ID1, channel1],[call_ID2, channel2]]
        for call in calls:
            if encoding == 'bbn':
                recreated_fname = 'fe_03_' + call[0] + '.txo'
                if int(call[0]) < 5851: # Fisher pt. 1: call IDs <= 05850
                    directory = dir1
                else: #Fisher pt. 2: call IDs > 05850
                    directory = dir2
                for folder in os.listdir(directory):
                    folder_path = os.path.join(directory, folder)
                    full_folder_path = os.path.join(folder_path, 'originals/') 
                    for filename in os.listdir(full_folder_path): 
                        if filename == recreated_fname:
                            fname = os.path.join(full_folder_path, filename)
                            if trunc_style == 'none':
                                speaker_lines = get_speaker_lines(fname, encoding, call[1])
                            elif trunc_style == 'beginning':
                                speaker_lines = get_speaker_lines_trunc_beg(fname, encoding, 
                                                                            call[1], trunc_size)
                            if speaker_lines: # make sure speaker has (enough) lines
                                transcripts.append(speaker_lines)
                            break 
                    else:
                        continue
                    break
            elif encoding == 'ldc':
                recreated_fname = 'fe_03_' + call[0] + '.txt'
                if int(call[0]) < 5851: # Fisher pt. 1: call IDs <= 05850
                    directory = dir1
                else: #Fisher pt. 2: call IDs > 05850
                    directory = dir2
                for folder in os.listdir(directory): 
                    folder_path = os.path.join(directory, folder) 
                    for filename in os.listdir(folder_path): 
                        if filename == recreated_fname:
                            fname = os.path.join(folder_path, filename)
                            if trunc_style == 'none':
                                speaker_lines = get_speaker_lines(fname, 'ldc', call[1])
                            elif trunc_style == 'beginning':
                                speaker_lines = get_speaker_lines_trunc_beg(fname, 'ldc', 
                                                                            call[1], trunc_size)
                            if speaker_lines: # make sure speaker has (enough) lines
                                transcripts.append(speaker_lines)
                            break
                    else:
                        continue
                    break
        if len(transcripts) == 2: # make sure BBN transcript exists for both calls
            neg_transcript = {'label': 0,
                            'call 1': transcripts[0],
                            'call 2': transcripts[1]}
            neg_transcripts.append(neg_transcript)
            neg_trials_final.append(trial)
    output_to_file(neg_transcripts, transcripts_outfile) # trials with transcripts
    print(f'Number {trial_type} trials: ', len(neg_transcripts))
    return neg_trials_final, len(neg_transcripts)


def get_speaker_lines(fname, encoding, channel):
    """ Get the speaker's utterances from the transcript file """
    if encoding == 'bbn':
        if channel == 'A':
            pattern = 'L: ' 
        elif channel == 'B':
            pattern = 'R: '
    elif encoding == 'ldc':
        if channel == 'A':
            pattern = 'A: '
        elif channel == 'B':
            pattern = 'B: '  
    speaker_lines = []
    with open(fname, encoding='cp1252') as f:    
        lines = f.read().splitlines()
    for line in lines: 
        if pattern in line:
            splitline = re.split(pattern, line)
            line_raw = splitline[1]
            line_encode = line_raw.encode("ascii", "ignore") 
            line_decode = line_encode.decode()
            speaker_lines.append(line_decode.strip()) 
    return speaker_lines


def get_speaker_lines_trunc_beg(fname, encoding, channel, trunc_size):
    """ Get speaker's utterances (first 5 removed) from transcript file """
    if encoding == 'bbn':
        if channel == 'A':
            pattern = 'L: ' 
        elif channel == 'B':
            pattern = 'R: '
    elif encoding == 'ldc':
        if channel == 'A':
            pattern = 'A: '
        elif channel == 'B':
            pattern = 'B: '   
    speaker_lines = []
    with open(fname, encoding='cp1252') as f:    
        lines = f.read().splitlines()
    count = 0
    for line in lines: 
        if pattern in line:
            if count >= trunc_size:
                splitline = re.split(pattern, line)
                line_raw = splitline[1]
                line_encode = line_raw.encode("ascii", "ignore") 
                line_decode = line_encode.decode()
                speaker_lines.append(line_decode.strip()) 
            else:
                count += 1
    return speaker_lines


###############################       Save and output       ###############################

def output_to_file(data, data_outfile): 
    np.save(data_outfile, data, allow_pickle=True)
    return


def output_json(data, data_outfile): 
    with open(data_outfile, 'wt') as writer:
        writer.write(json.dumps(data, indent=4))
    return
    

def output_txt(encoding, dataset, trials_counts, stats_outfile): 
    """ Output dataset stats to txt file """
    with open(stats_outfile, 'w') as o_f:
        o_f.write('Encoding: %s\n' % encoding)
        o_f.write('Dataset: %s' % dataset)
        o_f.write('\n---------------------------\n')
        for trial in trials_counts:
            o_f.write('Num %s trials: %i\n' % (trial['trial type'], trial['count']))
        o_f.write('\n---------------------------\n')
    return


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    fisher_dir1 = cfg['fisher_dir1']
    fisher_dir2 = cfg['fisher_dir2']
    trials_dir = cfg['trial_data_dir']
    stats_dir = cfg['trial_stats_dir']
    datasets = cfg['datasets']
    trial_types = cfg['trial_types']
    trunc_style = cfg['trunc_style']
    trunc_size = cfg['trunc_size']

    ### Fisher data folders
    bbn_dir1 = os.path.join(fisher_dir1, "data/bbn_orig/")
    bbn_dir2 = os.path.join(fisher_dir2, "data/bbn_orig/")
    ldc_dir1 = os.path.join(fisher_dir1, "data/trans/")
    ldc_dir2 = os.path.join(fisher_dir2, "data/trans/")

    ### Step 1: Get BBN transcripts for each trial
    for dataset in datasets: #'train', 'val', 'test' 
        bbn_trials_counts = []
        for trial_type in trial_types: #'manypos', 'manyneg', 'hardpos', 'hardneg', 'harderneg'
            ### Infile
            trials_info_file = os.path.join(trials_dir, f'{dataset}_{trial_type}_trials_info.json')

            ### Outfiles 
            trials_final_outfile = os.path.join(trials_dir, 
                                                f'{dataset}_{trial_type}_trials_info_final.json')
            bbn_transcripts_outfile = os.path.join(trials_dir, 
                                                   f'bbn_{dataset}_{trial_type}_trials.npy')
            
            ### Retrieve BBN transcripts for trials
            tic = time.perf_counter()
            print(f"Retrieving {dataset} set BBN transcripts for {trial_type} trials...")
            if 'pos' in trial_type:
                trials_final, trials_count = get_pos_transcripts('bbn', trial_type, 
                                                                 trials_info_file, trunc_style, trunc_size, bbn_dir1, bbn_dir2, bbn_transcripts_outfile)
            elif 'neg' in trial_type:
                trials_final, trials_count = get_neg_transcripts('bbn', trial_type, 
                                                                 trials_info_file, trunc_style, trunc_size, bbn_dir1, bbn_dir2, bbn_transcripts_outfile)
            bbn_trials_counts.append({'trial type': trial_type, 'count': trials_count})

            ### Output final trials list (after checking for BBN transcript) for input to LDC
            output_json(trials_final, trials_final_outfile) 
            toc = time.perf_counter()
            print(f"Finished retrieving {dataset} set BBN transcripts for {trial_type} trials in \
                  {(toc - tic)/60:0.4f} minutes")
        
        ### Output BBN trial stats to txt file
        bbn_stats_outfile = os.path.join(stats_dir, f'{dataset}_trials_stats_final_bbn.txt')
        output_txt('BBN', dataset, bbn_trials_counts, bbn_stats_outfile)
    
    ### Step 2: Get LDC transcripts for each trial
    for dataset in datasets: #'train', 'val', 'test' 
        ldc_trials_counts = []
        for trial_type in trial_types: #'manypos', 'manyneg', 'hardpos', 'hardneg', 'harderneg'
            ### Infile: final info file (to ensure BBN transcripts existed for each call)
            trials_final_file = os.path.join(trials_dir, 
                                             f'{dataset}_{trial_type}_trials_info_final.json') 
            
            ### Outfile
            ldc_transcripts_outfile = os.path.join(trials_dir, 
                                                   f'ldc_{dataset}_{trial_type}_trials.npy')

            ### Retrieve LDC transcripts for trials
            tic = time.perf_counter()
            print(f"Retrieving {dataset} set LDC transcripts for {trial_type} trials...")
            if 'pos' in trial_type:
                trials_final, trials_count = get_pos_transcripts('ldc', trial_type, 
                                                                 trials_final_file, trunc_style, trunc_size, ldc_dir1, ldc_dir2, ldc_transcripts_outfile)
            elif 'neg' in trial_type:
                trials_final, trials_count = get_neg_transcripts('ldc', trial_type, 
                                                                 trials_final_file, trunc_style, trunc_size, ldc_dir1, ldc_dir2, ldc_transcripts_outfile)
            ldc_trials_counts.append({'trial type': trial_type, 'count': trials_count})
            toc = time.perf_counter()
            print(f"Finished retrieving {dataset} set LDC transcripts for {trial_type} trials in \
                  {(toc - tic)/60:0.4f} minutes")
        
        ### Output LDC trial stats to txt file (should match BBN stats file)
        ldc_stats_outfile = os.path.join(stats_dir, f'{dataset}_trials_stats_final_ldc.txt')
        output_txt('LDC', dataset, ldc_trials_counts, ldc_stats_outfile)
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)



