"""
    To get positive and negative trials for each dataset (train, val, test) 
    Inputs: the following 2 files for x = train, val, test 
        x_pos_call_info.json
        x_all_call_info.json
    Outputs: 
        Positive trials (balanced according to restrictions)
        Hard positive trials (balanced according to restrictions)
        Negative trials (balanced according to restrictions)
        Hard negative trials (balanced according to restrictions)
        Harder negative trials (balanced according to restrictions)
"""

import json
import copy
import time
import random
from collections import Counter 
import yaml
import sys

###############################       Match positive trials       ###################################

def match_manypos_trials(f_max, m_max, topic_max, trials_max, pos_call_info_file, 
                        manypos_trials_outfile): 
    """ 
    Positive trials = same speaker, different call 
    NOTE can be restricted by choice of f_max, m_max, topic_max, trials_max

    Input data: 
        pos_call_data = {pin: [[gender, call_ID, channel, topic], [...]], ...}
    Output (json): can balance for gender and topic
        pos_trials = [{'PIN': '#', 'call 1': [gender, call_ID, channel, topic], 
                    'call 2': [gender, call_ID, channel, topic]}, 
                    {...}, ...]
    """
    with open(pos_call_info_file, 'r') as f:
        pos_call_data = json.loads(f.read())
    manypos_trials = [] 
    gender_count = {'f': 0, 'm': 0} # total f/m appearances regardless of speaker 
    topic_appearances = {} 
    speaker_appearances = {} 
    for key, value in pos_call_data.items(): 
        ky = copy.deepcopy(key) 
        val = copy.deepcopy(value) 
        for i in range(len(val)):  
            for j in range(i+1, len(val)):
                if (val[i][0] == 'f' and gender_count['f'] < f_max) \
                        or (val[i][0] == 'm' and gender_count['m'] < m_max): 
                    topic_appearances.setdefault(val[i][3], 0) # initialize topic entry if DNE yet
                    topic_appearances.setdefault(val[j][3], 0) 
                    if topic_appearances[val[i][3]] < topic_max \
                            and topic_appearances[val[j][3]] < topic_max: 
                        pos_trial = {'PIN': ky,
                                    'call 1': val[i],
                                    'call 2': val[j]}
                        manypos_trials.append(pos_trial)
                        gender_count[val[i][0]] += 1
                        topic_appearances[val[i][3]] += 1
                        topic_appearances[val[j][3]] += 1
                        speaker_appearances.setdefault(ky, 0) 
                        speaker_appearances[ky] += 1 
    print('manypos trials count: ', len(manypos_trials)) 
    print('manypos gender count: ', gender_count)
    print('manypos topics present: ', len(topic_appearances))
    topic_distribution = sorted(topic_appearances.items(), key=lambda trial: trial[1], reverse=True)
    print('manypos unique speakers: ', len(speaker_appearances))
    speaker_distribution = Counter(speaker_appearances.values())
    sorted_speaker_distr = dict(sorted(speaker_distribution.items(), key=lambda x:x[1], 
                                       reverse=True))
    stats_info = [len(manypos_trials), gender_count, len(topic_appearances), topic_distribution, 
                  len(speaker_appearances), sorted_speaker_distr]
    if type(trials_max) == int: # take a random sample of total trials
        sample = random.sample(manypos_trials, trials_max)
        print(len(sample))
        output_json(sample, manypos_trials_outfile)
    else:
        output_json(manypos_trials, manypos_trials_outfile)
    return stats_info


def match_hardpos_trials(max_count, trials_max, pos_call_info_file, hardpos_trials_outfile): 
    """ 
    Hard positive trials = same speaker, different call, DIFFERENT TOPIC
    NOTE can be restricted by choice of max_count, trials_max

    Input data: 
        pos_call_data = {pin: [[gender, call_ID, channel, topic], [...]], ...}
    Output (json): 
        hardpos_trials = [{'PIN': '#', 'call 1': [gender, call_ID, channel, topic], 
                        'call 2': [gender, call_ID, channel, topic]}, 
                        {...}, ...]
    """
    with open(pos_call_info_file, 'r') as f:
        pos_call_data = json.loads(f.read())
    hardpos_trials = [] 
    gender_count = {'f': 0, 'm': 0} # total f/m appearances regardless of speaker  
    topic_appearances = {} 
    speaker_appearances = {} 
    for key, value in pos_call_data.items(): 
        ky = copy.deepcopy(key)
        val = copy.deepcopy(value) 
        for i in range(len(val)):  
            for j in range(i+1, len(val)):
                topic_appearances.setdefault(val[i][3], 0) 
                topic_appearances.setdefault(val[j][3], 0) 
                if val[i][3] != val[j][3] and topic_appearances[val[i][3]] < max_count \
                        and topic_appearances[val[j][3]] < max_count: 
                    hardpos_trial = {'PIN': ky,
                                'call 1': val[i],
                                'call 2': val[j]}
                    hardpos_trials.append(hardpos_trial)
                    gender_count[val[i][0]] += 1
                    topic_appearances[val[i][3]] += 1
                    topic_appearances[val[j][3]] += 1
                    speaker_appearances.setdefault(ky, 0) 
                    speaker_appearances[ky] += 1 
    print('hardpos trials count: ', len(hardpos_trials)) 
    print('hardpos gender count: ', gender_count) 
    print('hardpos topics present: ', len(topic_appearances)) 
    topic_distribution = sorted(topic_appearances.items(), key=lambda trial: trial[1], reverse=True)
    print('hardpos unique speakers: ', len(speaker_appearances))
    speaker_distribution = Counter(speaker_appearances.values())
    sorted_speaker_distr = dict(sorted(speaker_distribution.items(), key=lambda x:x[1], 
                                       reverse=True))
    stats_info = [len(hardpos_trials), gender_count, len(topic_appearances), topic_distribution, 
                  len(speaker_appearances), sorted_speaker_distr]
    if type(trials_max) == int: # take a random sample of total trials
        sample = random.sample(hardpos_trials, trials_max)
        print(len(sample))
        output_json(sample, hardpos_trials_outfile)
    else:
        output_json(hardpos_trials, hardpos_trials_outfile)
    return stats_info


###############################       Match Negative Pairs       ###################################

def match_manyneg_trials(all_call_info_file, call_match_max, topic_max, trials_max, 
                        manyneg_trials_outfile): 
    """ 
    Negative trials = different speakers (same or different call)
    NOTE can be restricted by choice of call_match_max, topic_max, trials_max

    Input data: all_call_data = [[pin, gender, call_ID, channel, topic], ...]
    Outputs (json):
        manyneg_trials_info = [ [[pin, gender, call_ID, channel, topic],
                                [pin, gender, call_ID, channel, topic]], 
                               [ [],[] ], ...]
    """
    with open(all_call_info_file, 'r') as f:
        all_call_data = json.loads(f.read())
    manyneg_trials = [] 
    gender_count = {'f': 0, 'm': 0} # total f/m appearances regardless of speaker 
    topic_appearances = {} 
    speaker_appearances = {} 
    for i in range(len(all_call_data)):
        for j in range(i + 1, i + call_match_max): # limit each CALL to being trialed a max of x (arbitrary) times but a speaker might appear in more if they participated in multiple calls 
            if j < len(all_call_data): # prevents 'IndexError: list index out of range' 
                if all_call_data[i][0] != all_call_data[j][0]: # speakers are different
                    topic_appearances.setdefault(all_call_data[i][4], 0) 
                    topic_appearances.setdefault(all_call_data[j][4], 0) 
                    if topic_appearances[all_call_data[i][4]] < topic_max \
                            and topic_appearances[all_call_data[j][4]] < topic_max: 
                        manyneg_trials.append([all_call_data[i], all_call_data[j]])
                        gender_count[all_call_data[i][1]] += 1
                        gender_count[all_call_data[j][1]] += 1
                        topic_appearances[all_call_data[i][4]] += 1
                        topic_appearances[all_call_data[j][4]] += 1
                        speaker_appearances.setdefault(all_call_data[i][0], 0) 
                        speaker_appearances.setdefault(all_call_data[j][0], 0)
                        speaker_appearances[all_call_data[i][0]] += 1
                        speaker_appearances[all_call_data[j][0]] += 1
    print('manyneg trials count: ', len(manyneg_trials)) 
    print('manyneg gender count: ', gender_count) 
    print('manyneg topics present: ', len(topic_appearances)) 
    topic_distribution = sorted(topic_appearances.items(), key=lambda trial: trial[1], reverse=True)
    print('manyneg unique speakers: ', len(speaker_appearances))
    speaker_distribution = Counter(speaker_appearances.values())
    sorted_speaker_distr = dict(sorted(speaker_distribution.items(), key=lambda x:x[1], 
                                       reverse=True))
    stats_info = [len(manyneg_trials), gender_count, len(topic_appearances), topic_distribution, 
                  len(speaker_appearances), sorted_speaker_distr]
    if type(trials_max) == int: # take a random sample of total trials
        sample = random.sample(manyneg_trials, trials_max)
        print(len(sample))
        output_json(sample, manyneg_trials_outfile)
    else:
        output_json(manyneg_trials, manyneg_trials_outfile)
    return stats_info


def match_hardneg_trials(all_call_info_file, call_match_max, trials_max, hardneg_trials_outfile): 
    """ 
    Hard negative trials = different speakers, SAME TOPIC (same or different call)
    NOTE can be restricted by choice of call_match_max, trials_max

    Input data: 
        all_call_data = [[pin, gender, call_ID, channel, topic], ...]
    Outputs (json):
        hardneg_trials_info = [ [[pin, gender, call_ID, channel, topic],
                                [pin, gender, call_ID, channel, topic]], 
                               [ [],[] ], ...]
    """
    with open(all_call_info_file, 'r') as f:
        all_call_data = json.loads(f.read())
    hardneg_trials = [] 
    gender_count = {'f': 0, 'm': 0} # total f/m appearances regardless of speaker  
    topic_appearances = {} 
    speaker_appearances = {} 
    for i in range(len(all_call_data)):
        for j in range(i + 1, i + call_match_max): # limit each CALL being trialed max of x times
            if j < len(all_call_data): # prevents 'IndexError: list index out of range' 
                if all_call_data[i][0] != all_call_data[j][0] \
                        and all_call_data[i][4] == all_call_data[j][4]: # speakers are different AND topic is the same
                    hardneg_trials.append([all_call_data[i], all_call_data[j]])
                    gender_count[all_call_data[i][1]] += 1
                    gender_count[all_call_data[j][1]] += 1
                    topic_appearances.setdefault(all_call_data[i][4], 0) 
                    topic_appearances.setdefault(all_call_data[j][4], 0) 
                    topic_appearances[all_call_data[i][4]] += 1
                    topic_appearances[all_call_data[j][4]] += 1
                    speaker_appearances.setdefault(all_call_data[i][0], 0) 
                    speaker_appearances.setdefault(all_call_data[j][0], 0)
                    speaker_appearances[all_call_data[i][0]] += 1
                    speaker_appearances[all_call_data[j][0]] += 1
    print('hardneg trials count: ', len(hardneg_trials)) 
    print('hardneg gender count: ', gender_count)
    print('hardneg topics present: ', len(topic_appearances))
    topic_distribution = sorted(topic_appearances.items(), key=lambda trial: trial[1], reverse=True)
    print('hardneg unique speakers: ', len(speaker_appearances))
    speaker_distribution = Counter(speaker_appearances.values())
    sorted_speaker_distr = dict(sorted(speaker_distribution.items(), key=lambda x:x[1], 
                                       reverse=True))
    stats_info = [len(hardneg_trials), gender_count, len(topic_appearances), topic_distribution, 
                  len(speaker_appearances), sorted_speaker_distr]
    if type(trials_max) == int: # take a random sample of total trials
        sample = random.sample(hardneg_trials, trials_max)
        print(len(sample))
        output_json(sample, hardneg_trials_outfile)
    else:
        output_json(hardneg_trials, hardneg_trials_outfile)
    return stats_info


def match_harderneg_trials(all_call_info_file, trials_max, harderneg_trials_outfile): 
    """ 
    Harder negative trials = different speakers, same topic, SAME CALL 
    
    Input data: 
        all_call_data = [[pin, gender, call_ID, channel, topic], ...]
    Output (json):
        harderneg_trials_info = [ [[pin, gender, call_ID, channel, topic],
                                  [pin, gender, call_ID, channel, topic]], 
                                 [ [],[] ], ...]
    """
    with open(all_call_info_file, 'r') as f:
        all_call_data = json.loads(f.read())
    harderneg_trials = [] 
    gender_count = {'f': 0, 'm': 0} # total f/m appearances regardless of speaker  
    topic_appearances = {} 
    speaker_appearances = {} 
    for i in range(len(all_call_data)):
        for j in range(i + 1, len(all_call_data)): # limit each CALL being trialed max of x times
            if all_call_data[i][0] != all_call_data[j][0] \
                    and all_call_data[i][2] == all_call_data[j][2]: # speakers are different AND on the same call 
                harderneg_trials.append([all_call_data[i], all_call_data[j]])
                gender_count[all_call_data[i][1]] += 1
                gender_count[all_call_data[j][1]] += 1
                topic_appearances.setdefault(all_call_data[i][4], 0) 
                topic_appearances[all_call_data[i][4]] += 1
                speaker_appearances.setdefault(all_call_data[i][0], 0) 
                speaker_appearances.setdefault(all_call_data[j][0], 0)
                speaker_appearances[all_call_data[i][0]] += 1
                speaker_appearances[all_call_data[j][0]] += 1
    print('harderneg trials count: ', len(harderneg_trials)) 
    print('harderneg gender count: ', gender_count)
    print('harderneg topics present: ', len(topic_appearances))
    topic_distribution = sorted(topic_appearances.items(), key=lambda trial: trial[1], reverse=True)
    print('harderneg unique speakers: ', len(speaker_appearances))
    speaker_distribution = Counter(speaker_appearances.values())
    sorted_speaker_distr = dict(sorted(speaker_distribution.items(), key=lambda x:x[1], 
                                       reverse=True))
    stats_info = [len(harderneg_trials), gender_count, len(topic_appearances), topic_distribution, 
                  len(speaker_appearances), sorted_speaker_distr]
    if type(trials_max) == int: # take a random sample of total trials
        sample = random.sample(harderneg_trials, trials_max)
        print(len(sample))
        output_json(sample, harderneg_trials_outfile)
    else: 
        output_json(harderneg_trials, harderneg_trials_outfile)
    return stats_info



###############################       SAVE and OUTPUT       ###################################

def output_json(data, data_outfile): 
    with open(data_outfile, 'wt') as writer:
        writer.write(json.dumps(data, indent=4))
    return


def output_txt(set_type, f_max, m_max, topic_max_manypos, topic_max_manyneg, 
               call_match_max_manyneg, call_match_max_hardneg, trials_max_pos, trials_max_neg, trials_max_harder, stats_all, stats_outfile): 
    with open(stats_outfile, 'w') as o_f:
        o_f.write('%s stats\n' % set_type)
        o_f.write('Max num of females and males: %i, %i\n' % (f_max, m_max))
        o_f.write('Max times a topic can appear in manypos: %i\n' % topic_max_manypos)
        o_f.write('Max times a topic can appear in manyneg: %i\n' % topic_max_manyneg)
        o_f.write('Max times a call can be matched in manyneg: %i\n' % call_match_max_manyneg)
        o_f.write('Max times a call can be matched in hardneg: %i\n\n' % call_match_max_hardneg)
        o_f.write(f'Max number of pos trials sampled: {trials_max_pos}\n')
        o_f.write(f'Max number of neg trials sampled: {trials_max_neg}\n')
        o_f.write(f'Max number of harderneg trials sampled: {trials_max_harder}\n')
        for stat in stats_all:
            o_f.write('---------------------------\n')
            o_f.write('%s:\n' % stat['trial type'])
            o_f.write('Num trials: %i\n' % stat['stats'][0])
            o_f.write('Gender count: %s\n' % str(stat['stats'][1]))
            o_f.write('Num topics present: %i\n' % stat['stats'][2])
            o_f.write('Num unique speakers: %i\n' % stat['stats'][4])
            o_f.write('Topic appearances: %s\n' % str(stat['stats'][3]))
            o_f.write('Speaker appearances: %s\n\n' % str(stat['stats'][5]))
    return


###############################       MAIN       ###################################

def main(cfg):
    ### Get config parameters
    outdata_dir = cfg['outdata_dir']
    f_max = cfg['f_max']
    m_max = cfg['m_max'] 
    datasets = cfg['datasets']
    trial_types = cfg['trial_types']

    for set_type in datasets: #'train', 'val', 'test'
        ### Get dataset parameters
        topic_max_manypos = cfg[f'topic_max_manypos_{set_type}'] 
        topic_max_manyneg = cfg[f'topic_max_manyneg_{set_type}'] 
        call_match_max_manyneg = cfg[f'call_match_max_manyneg_{set_type}'] 
        call_match_max_hardneg = cfg[f'call_match_max_hardneg_{set_type}'] 
        trials_max_pos = cfg[f'trials_max_pos_{set_type}'] 
        trials_max_neg = cfg[f'trials_max_neg_{set_type}'] 
        trials_max_harder = cfg[f'trials_max_harder_{set_type}'] 
        
        ### Infiles
        pos_call_info_file = outdata_dir + f'{set_type}_pos_call_info.json' 
        all_call_info_file = outdata_dir + f'{set_type}_all_call_info.json'
        
        ### Create all trials for the dataset
        tic = time.perf_counter() 
        print(f"Starting {set_type} set trial creation...")
        stats_all = []
        for trial_type in trial_types: # 'manypos', 'manyneg', 'hardpos', 'hardneg', 'harderneg'
            ### Outfile for trials
            trials_outfile = outdata_dir + f'{set_type}_{trial_type}_trials_info.json' 
            
            ### Get trials by trial_type
            if trial_type == 'manypos': # same speaker, different call
                stats_manypos = match_manypos_trials(f_max, m_max, topic_max_manypos, 
                                                     trials_max_pos, pos_call_info_file, trials_outfile)  
                stats_all.append({'trial type': trial_type, 'stats': stats_manypos})
            elif trial_type == 'manyneg': # different speakers (same or different call)
                stats_manyneg = match_manyneg_trials(all_call_info_file, call_match_max_manyneg, 
                                                    topic_max_manyneg, trials_max_neg, trials_outfile) 
                stats_all.append({'trial type': trial_type, 'stats': stats_manyneg})
            elif trial_type == 'hardpos': # same speaker, different call, different topic
                stats_hardpos = match_hardpos_trials(topic_max_manypos, trials_max_pos, 
                                                    pos_call_info_file, trials_outfile) 
                stats_all.append({'trial type': trial_type, 'stats': stats_hardpos})
            elif trial_type == 'hardneg': # different speaker, same topic (same or different call)
                stats_hardneg = match_hardneg_trials(all_call_info_file, call_match_max_hardneg, 
                                                    trials_max_neg, trials_outfile) 
                stats_all.append({'trial type': trial_type, 'stats': stats_hardneg})
            elif trial_type == 'harderneg': # different speaker, same topic, SAME CALL
                stats_harderneg = match_harderneg_trials(all_call_info_file, trials_max_harder, 
                                                        trials_outfile) 
                stats_all.append({'trial type': trial_type, 'stats': stats_harderneg})
        
        ### Output dataset trial stats to txt file
        stats_outfile = outdata_dir + f'{set_type}_set_trials_stats.txt'
        output_txt(set_type.upper(), f_max, m_max, topic_max_manypos, topic_max_manyneg,      
                   call_match_max_manyneg, call_match_max_hardneg, trials_max_pos, trials_max_neg, trials_max_harder, stats_all, stats_outfile) 
        toc = time.perf_counter()
        print(f"Finished {set_type} set trial creation in {toc - tic:0.3f} seconds\n")
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)




