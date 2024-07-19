# SUB-GOAL DISTILLATION: A METHOD TO IMPROVE SMALL LANGUAGE AGENTS

This is the official repository for [SUB-GOAL DISTILLATION: A METHOD TO IMPROVE SMALL LANGUAGE AGENTS](https://arxiv.org/abs/2405.02749), accepted at CoLLAs 2024. Our code base is a modification and extension of the existing [SwiftSage](https://github.com/yuchenlin/SwiftSage) repository.


# Abstract

While Large Language Models (LLMs) have demonstrated significant promise as agents in interactive tasks, their substantial computational requirements and restricted number of calls constrain their practical utility, especially in long-horizon interactive tasks such as decision-making or in scenarios involving continuous ongoing tasks. To address these constraints, we propose a method for transferring the performance of an LLM with billions of parameters to a much smaller language model (770M parameters). Our approach involves constructing a hierarchical agent comprising a planning module, which learns through Knowledge Distillation from an LLM to generate sub-goals, and an execution module, which learns to accomplish these sub-goals using elementary actions. In detail, we leverage an LLM to annotate an oracle path with a sequence of sub-goals towards completing a goal. Subsequently, we utilize this annotated data to fine-tune both the planning and execution modules. Importantly, neither module relies on real-time access to an LLM during inference, significantly reducing the overall cost associated with LLM interactions to a fixed cost. In ScienceWorld, a challenging and multi-task interactive text environment, our method surpasses standard imitation learning based solely on elementary actions by 16.7% (absolute). Our analysis highlights the efficiency of our approach compared to other LLM-based methods. 


# Install

`pip install -r requirements.txt`

1- Generate dataset

2- Train models

3- Run agnet 
 
 parser.add_argument("--dir", default='/data/data_v4_all_first_sg/data_dir/test/')

        parser.add_argument("--lm_path", default= 'path to action generator model (executor)')
        parser.add_argument("--sg_lm_path", help='path to sg generator checkpoint model (controller)')
        parser.add_argument("--Fsg_lm_path", default='path to first sg checkpoint model') 


# Citation

`@inproceedings{
hashemzadeh2024sub,
title={Sub-goal Distillation: A Method to Improve Small Language Agents},
author={Hashemzadeh, Maryam and Stengel-Eskin, Elias and Chandar, Sarath and Cote, Marc-Alexandre},
booktitle={Third Conference on Lifelong Learning Agents (CoLLAs)},
year={2024},
url={https://arxiv.org/abs/2405.02749}
}`

