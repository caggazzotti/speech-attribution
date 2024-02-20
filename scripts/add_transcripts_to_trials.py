"""
    This script retrieves the transcript for each call in each set of positive and negative trials, combines the positive and negative trials into difficulty levels, and outputs the resulting trial files. Not all BBN transcripts exist, so the number of trials reduces after adding BBN transcripts to the trials. Adding BBN transcripts must be done first so the LDC trials match the BBN trials.

    Inputs: 
        Dataset files containing positive and negative trial info 
    Outputs: 
        Dataset files containing trials with corresponding BBN/LDC transcripts by difficulty level
            transcripts = [{'label': 0, 'call 1': ['utterance 1', 'utterance 2', '...'], 
                            'call 2': ['utterance 1', 'utterance 2', '...']}, 
                            {...}, ...]
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
                        dir1, dir2):
    """
    Retrieve transcripts for each call in the positive trials. Outputs a final list of trials info (to use in retrieving LDC transcripts) after checking the BBN transcript exists for each call.

    Input data:
        pos_trials_info = [{'PIN': '#', 'call 1': [gender, call_ID, channel, topic], 
                        'call 2': [gender, call_ID, channel, topic]}, 
                        {...}, ...]
    Returns:
        pos_trials_info_final = [{'PIN': '#', 'call 1': [gender, call_ID, channel, topic], 
                                'call 2': [gender, call_ID, channel, topic]}, 
                                {...}, ...]
        pos_transcript_trials = [{'label': 1, 'call 1': ['utterance 1', 'utterance 2', '...'], 
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
    print(f'Number {trial_type} trials: ', len(pos_transcripts))
    return pos_transcripts, pos_trials_final, len(pos_transcripts)


def get_neg_transcripts(encoding, trial_type, neg_trials_info_file, trunc_style, trunc_size, 
                        dir1, dir2):
    """
    Retrieve transcripts for each call in the negative trials. Outputs a final list of trials info (to use in retrieving LDC transcripts) after checking the BBN transcript exists for each call.

    Input data:
        neg_trials_info = [ [[pin, gender, call_ID, channel, topic],
                                [pin, gender, call_ID, channel, topic]], 
                                [ [],[] ], ...]
    Returns:
        neg_trials_info_final = [ [[pin, gender, call_ID, channel, topic],
                                [pin, gender, call_ID, channel, topic]], 
                                [ [],[] ], ...]
        neg_transcript_trials = [{'label': 0, 'call 1': ['utterance 1', 'utterance 2', '...'], 
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
    print(f'Number {trial_type} trials: ', len(neg_transcripts))
    return neg_transcripts, neg_trials_final, len(neg_transcripts)


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
            o_f.write('Difficulty: %s\n' % trial['difficulty'])
            o_f.write('Num positive trials: %i\n' % trial['pos trials count'])
            o_f.write('Num negative trials: %i\n' % trial['neg trials count'])
            o_f.write('Num total trials: %i\n\n' % trial['total trials count'])
        o_f.write('\n---------------------------\n')
    return


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    fisher_dir1 = cfg['fisher_dir1']
    fisher_dir2 = cfg['fisher_dir2']
    work_dir = cfg['work_dir']
    datasets = cfg['datasets']
    encodings = cfg['encodings']
    difficulties = cfg['difficulties']
    trunc_style = cfg['trunc_style']
    trunc_size = cfg['trunc_size']

    trials_dir = os.path.join(work_dir, 'trials_data')
    stats_dir = os.path.join(work_dir, 'trials_stats')

    ### Fisher data folders
    bbn_dir1 = os.path.join(fisher_dir1, "data/bbn_orig/")
    bbn_dir2 = os.path.join(fisher_dir2, "data/bbn_orig/")
    ldc_dir1 = os.path.join(fisher_dir1, "data/trans/")
    ldc_dir2 = os.path.join(fisher_dir2, "data/trans/")
    
    ### Step 1: Get BBN transcripts for each trial within each difficulty level
    for dataset in datasets: # 'train', 'val', 'test' 
        bbn_trials_counts = []
        for difficulty in difficulties: # 'base', 'hard', 'harder'
            if difficulty == 'base':
                trial_match = ['basepos', 'baseneg']
            elif difficulty == 'hard':
                trial_match = ['hardpos', 'hardneg']
            elif difficulty == 'harder':
                trial_match = ['hardpos', 'harderneg']
            
            ### Infiles
            pos_info_file = os.path.join(trials_dir, 
                                        f'{dataset}_{trial_match[0]}_trials_info.json')
            neg_info_file = os.path.join(trials_dir, 
                                        f'{dataset}_{trial_match[1]}_trials_info.json')

            ### Outfiles 
            pos_info_final_outfile = os.path.join(trials_dir, 
                                        f'{dataset}_{trial_match[0]}_trials_info_final.json')
            neg_info_final_outfile = os.path.join(trials_dir, 
                                        f'{dataset}_{trial_match[1]}_trials_info_final.json')
            
            ### Retrieve BBN transcripts for trials
            tic = time.perf_counter()
            print(f"Retrieving {dataset} set BBN transcripts for {difficulty} trials...")

            pos_transcripts, pos_trials_final, pos_trials_count = \
                get_pos_transcripts('bbn', trial_match[0], pos_info_file, trunc_style, trunc_size, 
                                    bbn_dir1, bbn_dir2)
            neg_transcripts, neg_trials_final, neg_trials_count = \
                get_neg_transcripts('bbn', trial_match[1], neg_info_file, trunc_style, trunc_size, 
                                    bbn_dir1, bbn_dir2)
            bbn_trials_counts.append({'difficulty': difficulty, 
                                    'pos trials count': pos_trials_count,
                                    'neg trials count': neg_trials_count,
                                    'total trials count': pos_trials_count + neg_trials_count})
            
            ### Output final trials after checking for BBN transcript (for matching LDC transcripts)
            output_json(pos_trials_final, pos_info_final_outfile) 
            output_json(neg_trials_final, neg_info_final_outfile) 

            ### Output BBN trials with transcripts by difficulty level
            bbn_transcripts_outfile = os.path.join(trials_dir, 
                                                f'bbn_{dataset}_{difficulty}_trials.npy')
            pos_neg_transcripts = pos_transcripts + neg_transcripts
            output_to_file(pos_neg_transcripts, bbn_transcripts_outfile) 

            toc = time.perf_counter()
            print(f"Retrieved trials in {(toc - tic)/60:0.4f} minutes")
        
        ### Output BBN trial stats to txt file
        bbn_stats_outfile = os.path.join(stats_dir, f'{dataset}_trials_stats_final_bbn.txt')
        output_txt('BBN', dataset, bbn_trials_counts, bbn_stats_outfile)
    
    ### Step 2: Get LDC transcripts for each trial
    if 'ldc' in encodings:
        for dataset in datasets: # 'train', 'val', 'test' 
            ldc_trials_counts = []
            for difficulty in difficulties: # 'base', 'hard', 'harder'
                if difficulty == 'base':
                    trial_match = ['basepos', 'baseneg']
                elif difficulty == 'hard':
                    trial_match = ['hardpos', 'hardneg']
                elif difficulty == 'harder':
                    trial_match = ['hardpos', 'harderneg']

                ### Infiles: final info files (to ensure BBN transcripts existed for each call)
                pos_final_file = os.path.join(trials_dir, 
                                            f'{dataset}_{trial_match[0]}_trials_info_final.json') 
                neg_final_file = os.path.join(trials_dir, 
                                            f'{dataset}_{trial_match[1]}_trials_info_final.json') 

                ### Retrieve LDC transcripts for trials
                tic = time.perf_counter()
                print(f"Retrieving {dataset} set LDC transcripts for {difficulty} trials...")
                pos_transcripts, pos_trials_final, pos_trials_count = \
                    get_pos_transcripts('ldc', trial_match[0], pos_final_file, trunc_style, 
                                        trunc_size, ldc_dir1, ldc_dir2)
                neg_transcripts, neg_trials_final, neg_trials_count = \
                    get_neg_transcripts('ldc', trial_match[1], neg_final_file, trunc_style, 
                                        trunc_size, ldc_dir1, ldc_dir2)
                ldc_trials_counts.append({'difficulty': difficulty, 
                                        'pos trials count': pos_trials_count,
                                        'neg trials count': neg_trials_count,
                                        'total trials count': pos_trials_count + neg_trials_count})

                ### Output LDC trials with transcripts by difficulty level
                ldc_transcripts_outfile = os.path.join(trials_dir, 
                                                    f'ldc_{dataset}_{difficulty}_trials.npy')
                pos_neg_transcripts = pos_transcripts + neg_transcripts
                output_to_file(pos_neg_transcripts, ldc_transcripts_outfile)
                
                toc = time.perf_counter()
                print(f"Retrieved trials in {(toc - tic)/60:0.4f} minutes")
            
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



