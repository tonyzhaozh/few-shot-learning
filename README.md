# Few-shot Learning With Language Models

This is a codebase to perform few-shot "in-context" learning using language models similar to the [GPT-3 paper](https://arxiv.org/abs/2005.14165). In particular, a few training examples are placed into a natural language "prompt" and predictions are made by generating from the language model. See the [GPT-3 paper](https://arxiv.org/abs/2005.14165) and [Calibrate Before Use](http://arxiv.org/abs/2102.09690) for more information.

You can run this codebase with GPT-3 (if you have a key from OpenAI), GPT-2, and any other language model available in [HuggingFace Transformers](https://huggingface.co/models). If you have a GPT-3 key, you should place your API key into a file named `openai_key.txt`. The underlying model you use is abstracted away using a common API.

Running this codebase will report results with and without [contextual calibration](http://arxiv.org/abs/2102.09690).

## Dependencies

This code is written using PyTorch and [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers). If you are running a model locally (e.g., GPT-2), the code requires a single GPU. Running these experiments is relatively lightweight (there is no training), so a single GPU is sufficient. It is technically possible to run the experiments without a GPU, but the runtime will be slow.

## Installation

The easiest way to install the code is to create a fresh anaconda environment:
```
conda create -n fewshot python=3.6
source activate fewshot
pip install -r requirements.txt
```
Now you should be ready to go!

## Replicating Our Results

Here is how to replicate the results from our paper for GPT-2. To replicate the results for classification tasks:
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py \
--model="gpt2-xl" \
--dataset="sst2, trec, cb, agnews, dbpedia" \
--num_seeds=5 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300 \
--approx
```

To replicate the results for extraction tasks:
```
CUDA_VISIBLE_DEVICES=0 python run_extraction.py \
--model="gpt2-xl" \
--dataset="mit_movie_Genre, mit_movie_Director, atis_airline_name, atis_depart_date.day_name" \
--num_seeds=5 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300
```

To replicate the results for LAMA:
```
CUDA_VISIBLE_DEVICES=0 python run_lama.py
```
Note that after we refactored our code, the training sets are not the same ones used in our results table. We expect the results to differ slightly but they should match the same trends seen in our results.

## Overview of Codebase

### Data
The `data` folder contains the raw data for numerous tasks. If you'd like to add your own task, add the data into that folder. The code for loading a dataset, as well as defining the prompt format for a task, is in `utils/data_utils.py`. We have loaders for a wide range of existing datasets. If you want to add a new dataset that is similar in structure to any of the existing datasets (e.g., its text classification) adding it should be very simple---you can use an existing dataset as a guide.

### Utils
The `utils` folder contains all of the code for calling the underlying models, getting the probabilities of each label token, possibly applying contextual calibration, and more. If you just want to evaluate few-shot learning on your task, you should not need to modify this code. If you want to extend our code (e.g., modify how decisions are made) this is the place to look.

### Run Scripts
The run scripts, e.g., `run_classification.py`, contain the code for randomly sampling the examples to use in the prompt, calling the models, the necessary evaluation metrics, and more. If you are adding a new task format (one that is not classification, QA) then you will need to write your own run script. Inside the run script, you can set the parameters for the experiments using the command line arguments.

For all experiments, we save and pickle the outputs of the model. This makes doing a post-hoc analysis of the accuracy / plotting results / etc. very fast. You can also use the saved outputs to evaluate how the accuracy would have changed if a different decision making function was used (e.g., accuracy with and without contextual calibration).


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@article{Zhao2021Calibrate,	
  Author = {Tony Z. Zhao and Eric Wallace and Shi Feng and Dan Klein and Sameer Singh},	
  Journal={arXiv preprint arXiv:2102.09690},	
  Year = {2021},	
  Title = {Calibrate Before Use: Improving Few-shot Performance of Language Models}	
}    	
```

## Contributions and Contact

This code was developed by Tony Z. Zhao and Eric Wallace, contact available at tonyzhao0824@berkeley.edu and ericwallace@berkeley.edu.	

If you'd like to contribute code, feel free to open a [pull request](https://github.com/tonyzhaozh/few-shot-learning/pulls). If you find an issue, please open an [issue](https://github.com/tonyzhaozh/few-shot-learning/issues).
