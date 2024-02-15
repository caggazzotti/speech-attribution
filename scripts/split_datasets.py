"""
    This script separates Fisher pt. 1 and pt. 2 into training, validation, and test datasets based on speaker so each set has unique speakers. The pin (speaker ID), gender, call_ID, topic, and channel is collected for all the calls and saved in pos_call_info.json for later positive pair creation and all_call_info.json for later negative pair creation.

    Inputs: pindata_tbl, calldata_tbl_pt1, calldata_tbl_pt2 (Fisher files)
    Outputs: following 4 dataset files for x = train, val, test 
        x_pin_IDs.json: ['1006', '10008',...]
        x_pos_call_info.json: {pin: [[gender, call_ID, channel, topic], [...]], ...}
        x_all_call_info.json: [[pin, gender, call_ID, channel, topic], ...]
        x_set_stats.txt 
"""

import re
import pandas as pd
import numpy as np
import json
import copy
import time
import yaml
import sys


def split_datasets(train_fraction, test_fraction, r_state, pindata_tbl):
    """
    Datasets divided based on pin/speaker (pin IDs range from 1006-99997) 
    pindata.tbl contains ALL pins/speakers across Fisher pt. 1 and pt. 2

    Inputs: 
        fractions of the whole dataset to reserve for training set and test set
        random state
        Fisher pindata table full of call records per PIN/speaker
    Outputs: 
        test_set array
        val_set array
        train_set array
    """
    pin_data = pd.read_table(pindata_tbl, delimiter=',', 
                             converters={'PIN': str}) # to keep leading zeros in PIN
    num_speakers = len(pin_data)
    print('# of total speakers:', num_speakers)
    
    test_split = train_fraction + test_fraction # np.split fractions are cumulative
    train_set, test_set, val_set = np.split(
        pin_data.sample(frac=1, random_state=r_state), 
        [int(train_fraction * len(pin_data)), int(test_split * len(pin_data))]) 
    print('train-val-test # of speakers:', len(train_set), len(val_set), len(test_set))
    return train_set, val_set, test_set, num_speakers


def get_pin_data(set_data, pin_outfile):
    """
    Get pin, call ID, gender, and channel info for each dataset
    Speaker calls can appear in both positive and negative trials

    Inputs: 
        respective array of pindata (train_set/val_set/test_set)
    Outputs: 
        pos_pin_data = {pin_ID: [[gender, call_ID, channel], [...]], pin_ID: [[]], ...} 
            only speakers who participated in multiple calls
        all_pin_data = [[pin, gender, call_ID, channel], [...], ...] 
            pin and call data for every speaker in Fisher 1 and 2
        (to json) pin_IDs = ['1006', '99004',...]
        stats_info with various sizes/distributions of the dataset
    """
    pos_pin_data = {}
    all_pin_data = []
    pin_IDs = []
    gender_count = {'f': 0, 'm': 0}
    for i in range(len(set_data)): 
        call_info = []
        pindata = set_data.iloc[i] 
        gender = pindata['S_SEX']
        pin_ID = pindata['PIN']
        if isinstance(gender, str): # skips NA (unknown gender)
            gender = gender.lower() 
            gender_count[gender] += 1 # count gender per speaker not per # of appearances
            pin_IDs.append(pin_ID) # for sanity checking no speaker overlap between sets
            calls = re.split(';', pindata['SIDE_DATA'])   
            for call in calls:
                if gender in call: # check genders match (only safeguard that same speaker picked up the phone)
                    call_ID = re.findall('(\d{5})', call)[0] 
                    channel = call[6:7] # A/B (LDC), L/R (BBN)
                    call_info.append([gender, call_ID, channel])
            if len(calls) > 1: # speaker participated in multiple calls
                pos_pin_data[pin_ID] = call_info
            info = copy.deepcopy(call_info) 
            for item in info:
                item.insert(0, pin_ID)
                all_pin_data.append(item) # for later negative pair creation
    stats_info = [len(set_data), len(pin_IDs), len(pos_pin_data), len(all_pin_data), gender_count]
    output_json(pin_IDs, pin_outfile)
    return pos_pin_data, all_pin_data, stats_info, pin_IDs 


def get_call_data(calldata_tbl_pt1, calldata_tbl_pt2, data, data_outfile, label):
    """ 
    Get topic info for each call and output to json
    calldata.tbl_pt1 (call IDs range from 00001-05850) and calldata.tbl_pt2 (05851-11699) 

    Input: either pos_pin_data or all_pin_data 
        pos_pin_data = {pin: [[gender, call_ID, channel], [gender, call_ID, channel]], ...}
        all_pin_data = [[pin, gender, call_ID, channel], ...]
    Ouput: same but with topic added and filtered for if topic exists 
        (to json) positive call_data = {pin: [[gender, call_ID, channel, topic], [...]], ...}
        (to json) all call_data = [[pin, gender, call_ID, channel, topic], ...]
        num of speakers (positive) or calls (all)
    """
    call_data_pt1 = pd.read_table(calldata_tbl_pt1, delimiter=',', 
                                  converters={'CALL_ID': str}) # keep leading zeros
    call_data_pt2 = pd.read_table(calldata_tbl_pt2, delimiter=',', 
                                  converters={'CALL_ID': str})
    if label == 'all':
        call_data = []
        for call in data:
            cl = copy.deepcopy(call)
            if int(cl[2]) < 5851: # Fisher pt. 1: call IDs <= 05850
                calldata = call_data_pt1.loc[call_data_pt1['CALL_ID'] == cl[2]] 
            else: # Fisher pt. 2: call IDs > 05850
                calldata = call_data_pt2.loc[call_data_pt2['CALL_ID'] == cl[2]] 
            topic = calldata['TOPICID']
            if topic.isnull().values.any() == False: # exclude empty topics
                cl.extend(topic)
                call_data.append(cl)
        output_json(call_data, data_outfile)
        print('num speakers: ', len(call_data))
    elif label == 'pos':
        call_data = {}
        for key, value in data.items(): 
            ky = copy.deepcopy(key) # so append/extend doesn't change dict every iteration
            val = copy.deepcopy(value) 
            call_items = []
            for call in val:
                if int(call[1]) < 5851: # Fisher pt. 1: call IDs <= 05850
                    calldata = call_data_pt1.loc[call_data_pt1['CALL_ID'] == call[1]] 
                else: # Fisher pt. 2: call IDs > 05850
                    calldata = call_data_pt2.loc[call_data_pt2['CALL_ID'] == call[1]]
                topic = calldata['TOPICID']
                if topic.isnull().values.any() == False: # exclude empty topics
                    call_items.append(call) 
                    call_items[-1].extend(topic) 
            if len(call_items) > 1: # speaker participated in multiple calls 
                call_data[ky] = call_items
        output_json(call_data, data_outfile)
        print('num calls: ', len(call_data))
    else:
        print('Invalid label')
        call_data = []
    return len(call_data)


def sanity_check_sets(train_pin_IDs, val_pin_IDs, test_pin_IDs):
    """ Make sure each set contains unique speakers"""

    assert len(set(train_pin_IDs)) == len(train_pin_IDs), \
        'Training set should not contain duplicate speakers'
    assert len(set(val_pin_IDs)) == len(val_pin_IDs), \
        'Validation set should not contain duplicate speakers'
    assert len(set(test_pin_IDs)) == len(test_pin_IDs), \
        'Test set should not contain duplicate speakers'
    assert bool(set(train_pin_IDs) & set(val_pin_IDs)) == False \
        ^ bool(set(train_pin_IDs) & set(test_pin_IDs)) == False \
        ^ bool(set(val_pin_IDs) & set(test_pin_IDs)) == False, \
            'Datasets should not have overlap'
    return


def output_json(data, data_outfile): 
    with open(data_outfile, 'wt') as writer:
        writer.write(json.dumps(data, indent=4))
    return


def output_txt(set_type, r_state, num_speakers, split_perc, num_pins, num_calls, \
               stats_info, stats_outfile): 
    with open(stats_outfile, 'w') as o_f:
        o_f.write('Total num speakers: %i\n' % num_speakers)
        o_f.write('Dataset split (train-val-test): %s\n' % split_perc)
        o_f.write('Random state: %i\n' % r_state)
        o_f.write('---------------------------\n')
        o_f.write('%s stats:\n' % set_type)
        o_f.write('Num speakers in initial dataset split: %i\n' % stats_info[0])
        o_f.write('Num speakers after gender check: %i\n' % stats_info[1])
        o_f.write('Num speakers for pos pairs before adding topic: %i\n' % stats_info[2])
        o_f.write('Num calls for neg pairs before adding topic: %i\n' % stats_info[3])
        o_f.write('Num speakers for pos pairs after adding topic: %i\n' % num_pins)
        o_f.write('Num calls for neg pairs after adding topic: %i\n' % num_calls)
        o_f.write('Gender: %s\n' % str(stats_info[4]))
    return


###############################       MAIN       ###################################
def main(cfg):
    ### Get config parameters
    fisher_dir1 = cfg['fisher_dir1']
    fisher_dir2 = cfg['fisher_dir2']
    outdata_dir = cfg['outdata_dir']
    r_state = cfg['r_state'] 
    test_fraction = cfg['test_fraction']
    train_fraction = cfg['train_fraction']
    
    ### Fisher data files
    calldata_tbl_pt1 = fisher_dir1 + 'fe_03_p1_calldata.tbl'
    calldata_tbl_pt2 = fisher_dir2 + 'fe_03_p2_calldata.tbl'
    pindata_tbl = fisher_dir1 + 'fe_03_pindata.tbl' 
    
    ### Split datasets by speaker
    val_fraction = 1 - train_fraction - test_fraction
    split_perc = f'{train_fraction}-{val_fraction}-{test_fraction}' # for output file
    train_set, val_set, test_set, num_speakers = split_datasets(train_fraction, \
                                                                test_fraction, r_state, pindata_tbl)
    dataset_splits = {'train': train_set, 'val': val_set, 'test': test_set}
    
    ### Get each datasets' info and output to json
    tic = time.perf_counter() 
    print(f"Starting all dataset collection...")
    all_pin_IDs = []
    for set_type, set_data in dataset_splits.items():
        ### Output files
        pinIDs_outfile = outdata_dir + f'{set_type}_pin_IDs.json'
        pos_data_outfile = outdata_dir + f'{set_type}_pos_call_info.json'
        all_data_outfile = outdata_dir + f'{set_type}_all_call_info.json'
        stats_outfile = outdata_dir + f'{set_type}_set_stats.txt'
        
        ### Get speaker info
        pos_pin_data, all_pin_data, stats_info, pin_IDs  = get_pin_data(set_data, pinIDs_outfile)
        all_pin_IDs.append(pin_IDs) # for later sanity check of no speaker overlap  
        
        ### Get call info
        print(f'{set_type} dataset---------')
        num_pins = get_call_data(calldata_tbl_pt1, calldata_tbl_pt2, 
                                 pos_pin_data, pos_data_outfile, 'pos')
        num_calls = get_call_data(calldata_tbl_pt1, calldata_tbl_pt2, 
                                  all_pin_data, all_data_outfile, 'all')
        output_txt(set_type, r_state, num_speakers, split_perc, num_pins, 
                   num_calls, stats_info, stats_outfile)
    toc = time.perf_counter()
    print(f"Collected all dataset info in {toc - tic:0.3f} seconds")

    ### Sanity check no speaker overlap between sets
    sanity_check_sets(all_pin_IDs[0], all_pin_IDs[1], all_pin_IDs[2]) 
    return


if __name__ == '__main__':
    try:
        yaml_path = sys.argv[1]
    except:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = yaml.safe_load(open(yaml_path)) 
    main(cfg)


