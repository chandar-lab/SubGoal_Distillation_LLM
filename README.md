# SubGoal_Distillation_LLM
Code for the paper Sub-goal Distillation: A Method to Improve Small Language Agents

The code will be uploaded soon. 


1- Generate dataset

2- Train models

3- Run agnet 
 
 parser.add_argument("--dir", default='/data/data_v4_all_first_sg/data_dir/test/')

        parser.add_argument("--lm_path", default= 'path to action generator model (executor)')
        parser.add_argument("--sg_lm_path", help='path to sg generator checkpoint model (controller)')
        parser.add_argument("--Fsg_lm_path", default='path to first sg checkpoint model') 
