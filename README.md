# SUB-GOAL DISTILLATION: A METHOD TO IMPROVE SMALL LANGUAGE AGENTS

This is the official repository for [SUB-GOAL DISTILLATION: A METHOD TO IMPROVE SMALL LANGUAGE AGENTS](https://arxiv.org/abs/2405.02749), accepted at CoLLAs 2024 (Oral). Our code base is a modification and extension of the existing [SwiftSage](https://github.com/yuchenlin/SwiftSage) repository.


# Abstract

While Large Language Models (LLMs) have demonstrated significant promise as agents in interactive tasks, their substantial computational requirements and restricted number of calls constrain their practical utility, especially in long-horizon interactive tasks such as decision-making or in scenarios involving continuous ongoing tasks. To address these constraints, we propose a method for transferring the performance of an LLM with billions of parameters to a much smaller language model (770M parameters). Our approach involves constructing a hierarchical agent comprising a planning module, which learns through Knowledge Distillation from an LLM to generate sub-goals, and an execution module, which learns to accomplish these sub-goals using elementary actions. In detail, we leverage an LLM to annotate an oracle path with a sequence of sub-goals towards completing a goal. Subsequently, we utilize this annotated data to fine-tune both the planning and execution modules. Importantly, neither module relies on real-time access to an LLM during inference, significantly reducing the overall cost associated with LLM interactions to a fixed cost. In ScienceWorld, a challenging and multi-task interactive text environment, our method surpasses standard imitation learning based solely on elementary actions by 16.7% (absolute). Our analysis highlights the efficiency of our approach compared to other LLM-based methods. 


# Install

```bash
pip install -r requirements.txt
```

# Usage
## 1- Generate dataset

Codes for generating dataset are in `/data`.

       - unzip data/goldpaths-all.zip
       - python data/data_convert.py
       - python data/read_file.py

## 2- Train models

Codes for training the models are in `/train`. Three models which are small LM required to be fine-tuned: 1- action generator (executor), 2- sub-goal generator (contoroller), 3- first sub-goal generator. 

To train each of them, run its `ds_train*.sh` bash file. Set `cache_dir` to a path for cache, `model_name` to the LM name in HF, and `output_dir` to a path to save the checkpoints

```bash
bash ds_train.sh $cache_dir $model_name $output_dir
bash ds_train_sg.sh $cache_dir $model_name $output_dir
bash ds_train_sg_first.sh $cache_dir $model_name $output_dir
```

## 3- Run agnet 


The agent is in `/evaluation`. To run it do:

```bash
python evaluation/agent.py --dir 'test_dataset_path' --lm_path 'path_to_executor' --sg_lm_path 'path_to_controller' --Fsg_lm_path 'path_to_first_sg_generator'
```



# Citation

```bash
@inproceedings{
hashemzadeh2024sub,
title={Sub-goal Distillation: A Method to Improve Small Language Agents},
author={Hashemzadeh, Maryam and Stengel-Eskin, Elias and Chandar, Sarath and Cote, Marc-Alexandre},
booktitle={Third Conference on Lifelong Learning Agents (CoLLAs)},
year={2024},
url={https://arxiv.org/abs/2405.02749}
}
```

# Acknowledgements

We thank the authors of SwiftSage repository, which this repo is based upon.
