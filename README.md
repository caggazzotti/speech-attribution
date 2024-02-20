# Attribution of Transcribed Speech

This is the official repository for the 2023 paper ["Can Authorship Attribution Models Distinguish Speakers
in Speech Transcripts?"](https://arxiv.org/abs/2311.07564). The paper presents a new benchmark for speaker attribution by applying authorship attribution methods to conversational speech transcripts. We establish the state of the art on this benchmark by comparing various neural and non-neural baselines, finding that written text attribution models perform relatively well in certain settings, but perform worse as conversational topic is increasingly controlled.

## Installation

To create an environment with the required packages, run the following commands within the speech-attribution directory:

	python3 -m venv speech_attr
	. ./speech_attr/bin/activate
	pip3 install -U pip
	pip3 install -r requirements.txt

## Setup
The absolute paths for the Fisher data and the working directory for this project need to be manually added by modifying the following path variables in `config.yaml`:

- fisher_dir1: directory containing Fisher pt. 1 data (placeholder: ./LDC2004T19)
- fisher_dir2: directory containing Fisher pt. 2 data (placeholder: ./LDC2005T19)
- work_dir: where the trial datasets and results will be stored (placeholder: ./speech-attribution)

## Creating the trials and model embeddings

The parameters for all of the subsequent steps are contained in the `config.yaml` file. The settings used in the paper are the defaults so the results can be reproduced, but the parameters can easily be adjusted.

This pipeline is separated into chronological, incremental steps to provide explainability and sanity checks as well as flexibility for adapting to other datasets. As a result, various data and stats files are incrementally output throughout the pipeline to a `trials_data` and `trials_stats` directory, respectively.


### Step 1: Split datasets
Splits the entire Fisher corpus (parts 1 and 2) by speaker into training, validation, test datasets. The default is 50%-25%-25%.

	python scripts/split_datasets.py config.yaml

### Step 2: Create trials
For each dataset (train, val, test) and difficulty's (base, hard, harder) trial type (positive, negative), create trials, or pairs of calls matched together based on the restrictions set in `config.yaml`.

	python scripts/create_trials.py config.yaml

### Step 3: Add transcripts to trials
Retrieve the Fisher transcript for each trial of each trial type (positive, negative) within each difficulty level (base, hard, harder). First, the BBN transcripts are retrieved to ensure that the BBN transcript for each call in each trial exists. Then the corresponding LDC transcripts for those trials with an existing BBN transcript are retrieved so the trials for both encodings are the same. Finally, each difficulty's trial types are combined together (combine the positive and negative trials for each difficulty level) to create the final transcript trials by difficulty.

Note that following the findings in the paper, each speaker's first five utterances in each transcript are excluded to remove identifying speaker and conversational topic assignment information that could unfairly advantage the models in speaker verification. However, this parameter can be changed using `trunc_style` and `trunc_size` (# utterances) in the `config.yaml` file.

	python scripts/add_transcripts_to_trials.py config.yaml

### Step 4: Add model embeddings to transcript trials 
Using the final transcript trials from the previous step, use the respective model to embed each call within each trial (across encodings, datasets, and difficulties). Each model is run separately by the scripts below. A new directory is automatically created for each model that will contain the embedded trials and the later training and evaluation outputs.

	python scripts/embed_trials_SBERT.py config.yaml
	python scripts/embed_trials_CISR.py config.yaml
	python scripts/embed_trials_LUAR.py config.yaml


## Training and Evaluating the Models
The `config.yaml` file provides flexibility in evaluating on the validation or test set (`eval_type`) and choosing the hyperparameters for training (e.g. solver, max iterations) and evaluating (i.e. number of resamples for bootstrapping).The out-of-the-box models can be directly evaluated using `evaluate_models.py`, but the fine-tuned models require an additional prior training step before evaluation.

### Train the classifier (for fine-tuned models)
Train an MLP classifier on the training set trial embeddings for each model, encoding, and difficulty.

	python scripts/train_ft_models.py config.yaml

### Evaluate the out-of-the-box models and fine-tuned models
The out-of-the-box and fine-tuned models can be evaluated separately by setting the `model_versions` parameter in `config.yaml` to 'o' for out-of-the-box and 'ft' for fine-tuned.

	python scripts/evaluate_models.py config.yaml


## Citation
If you use our benchmark in your work, please consider citing our paper:

	@misc{speech-aa2023,
      title={Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?}, 
      author={Cristina Aggazzotti and Nicholas Andrews and Elizabeth Allyn Smith},
      year={2023},
      eprint={2311.07564},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}