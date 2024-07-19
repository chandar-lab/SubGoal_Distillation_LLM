import copy
import json
import os
from data_utils import add_current_place, add_current_objects, downsampling, get_real_task_id, compose_instance_v4_sg, compose_instance_v4_sgaslabel
import argparse
from collections import defaultdict, Counter
from subgoals.sg_generating import sg_generator

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default = 'fast_system', help = 'mode of data')
parser.add_argument('--dir', type = str, default = './data', help = 'directory to store data')
parser.add_argument('--lite', action = 'store_true')
parser.add_argument('--uniform', action = 'store_true')
parser.add_argument('--data_split', action = 'store_true', help = 'split subtasks into train/test subtask')
parser.add_argument('--task_id', type = str, default = '29', help = 'the task id')
parser.add_argument('--labeling', type = str, default = 'sg', help = 'indicate whether the output is actions or subgoals, it would be either <sg> or <action>')

args = parser.parse_args()

# update variables based on arguments
data_split = args.data_split
K = 10

# load the goal trajectories
gold_data_path = 'goldsequences-0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29.json'
with open(gold_data_path, 'r') as f:
    raw_data = json.load(f) 

data = []
train_data = []
val_data = []
test_data = []
train_data_by_task = defaultdict(list)
val_data_by_task = defaultdict(list)
test_data_by_task = defaultdict(list)
training_task_ids = raw_data.keys()
all_actions = []
task_idx_real_distribution = []
task_id_to_vars = {}
task_id_to_actions = {}
task_id = args.task_id

for i in range(1):    
    curr_task = raw_data[task_id]
    task_name = curr_task["taskName"]
    task_idx = curr_task["taskIdx"]
    if task_name.startswith("task"):  
        second_index = task_name.index('-', task_name.index('-') + 1)
        task_name = task_name[second_index+1:]
        task_name = task_name.replace("(","")
        task_name = task_name.replace(")","") 
    print(task_name)
    task_idx_real = get_real_task_id(task_name)
    task_group_id = task_idx_real.split("-")[0]
    curr_task_seq = curr_task['goldActionSequences']
     
    curr_task_seq = downsampling(task_idx_real, curr_task_seq)

    print(f"task_id: {task_id}; task_idx: {task_idx}; task_idx_real: {task_idx_real}")
    print(f"Task name: {task_name};  #Vars: {len(curr_task_seq)}")
    task_id_to_vars[task_id] = {"train":0, "dev": 0, "test": 0}
    task_id_to_actions[task_id] = {"train":0, "dev": 0, "test": 0}

    
    ### Start data processing
    fold_test_count = 0
    fold_dev_count = 0
    
    for seq_sample in curr_task_seq:
        task_desc = seq_sample['taskDescription']
        VarId = seq_sample['variationIdx']
        print("VarId is", VarId)
        places = []
        objects = [] 

        original_steps = seq_sample['path']

        steps = []
        for s in original_steps:
            if s['action'] == "look around":
                continue
            if s["action"].startswith("close door"):
                continue
            steps.append(s)
 
        if len(steps) < 2:
            continue
        fold = seq_sample['fold']
        if fold == 'test':
            fold_test_count += 1
        if fold_test_count == 10:
            break
        
        task_id_to_vars[task_id][fold] += 1        
        obs = steps[0]['observation']
        action = steps[0]['action']
        gold_length = len(steps) 
        gold_path_only = []

        for item in steps:
            gold_path_only.append(item['action'])
        
        task_name_id  = task_name + '_' + task_id
        steps,  subgoals_list_insteps = sg_generator(gold_path_only, task_desc, VarId, steps, curr_task_seq[:2], task_name_id)
        subgoal_cnt = 0
        subgoals = []
        subgoals.append(subgoals_list_insteps[subgoal_cnt])

        for i in range(1, len(steps)): 
            
            if i >= 2:
                recent_actions.append(steps[i-1]['action'])
                recent_obs.append(steps[i-1]['observation'])
                recent_scores.append(float(steps[i-1]['score']))
                recent_reward.append(recent_scores[-1]-recent_scores[-2])
            else: 
                recent_actions = ["look around"]
                recent_obs = ["N/A"]
                recent_scores = [0]
                recent_reward = [0]

            prev_step = steps[i - 1]
            curr_step = steps[i]

            ## find the subgoal of the next step
            if i+1 < len(steps):
                next_step = steps[i+1]
                next_subgoal = next_step['subgoal']
            else:
                next_subgoal = 'complete_task()'


            prev_prev_step = steps[i - 2] if i >= 2 else None

            returns_to_go = 1.0 - float(prev_step['score'])
            returns_to_go = round(returns_to_go, 2)

            prev_action = prev_step['action']
            curr_action = curr_step['action']
            prev_obs = prev_prev_step['observation'] if i >= 2 else "N/A"
            curr_obs = prev_step['observation']
            look = curr_step['freelook']
            prev_look = prev_step['freelook']
            inventory = curr_step['inventory']

            # add subgoals
            if curr_step['subgoal'] != subgoals[subgoal_cnt]:
                print('steps goal is: ', curr_step['subgoal'], ' --- sg from the array is :' , subgoals[subgoal_cnt])
                subgoal_cnt +=1
                subgoals.append(subgoals_list_insteps[subgoal_cnt])


            # Extract current place
            add_current_place(curr_obs, look, places)

            # Extract objects
            add_current_objects(task_id, look, objects, limit=25)


            if args.labeling== 'action':
                input_str, label = compose_instance_v4_sg(i, task_desc, returns_to_go, curr_action,
                                        curr_obs, inventory, look, prev_action, prev_obs,
                                        objects, places,
                                        recent_actions, recent_obs, recent_scores, recent_reward, subgoals)

            elif args.labeling == 'sg':
                input_str, label = compose_instance_v4_sgaslabel(i, task_desc, returns_to_go, curr_action,
                                                          curr_obs, inventory, look, prev_action, prev_obs,
                                                          objects, places,
                                                          recent_actions, recent_obs, recent_scores, recent_reward,
                                                          subgoals, next_subgoal)
            else:
                assert False, " ----- The labeling is not indicated! -----"

            curr_dat = {'input': input_str, 'target': label}
            
            curr_dat["task_id"] = int(task_id)
            curr_dat["variation_id"] = VarId
            curr_dat["task_real_id"] = task_idx_real
            task_idx_real_distribution.append(task_idx_real)
            all_actions.append(label)

            task_id_to_actions[task_id][fold] += 1
            
            if fold == 'train':
                train_data.append(curr_dat)
                train_data_by_task[task_group_id].append(curr_dat)
            elif fold == 'dev':
                val_data.append(curr_dat)
                val_data_by_task[task_group_id].append(curr_dat)
            elif fold == 'test':
                test_data.append(curr_dat)
                test_data_by_task[task_group_id].append(curr_dat)
 
 
## split the data in train, test, and validation.
for tid, v_stat in task_id_to_vars.items():
    print(f'{tid}, {v_stat["train"]}, {v_stat["dev"]}, {v_stat["test"]}')

for tid, v_stat in task_id_to_actions.items():
    print(f'{tid}, {v_stat["train"]}, {v_stat["dev"]}, {v_stat["test"]}')

counter = Counter(task_idx_real_distribution)
for value, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{value}: {count}")

action_counter = Counter([a.split()[0] for a in all_actions])
for value, count in sorted(action_counter.items(), key=lambda x: x[1], reverse=True):
    print(f"{value}: {count}")

## Save the data
if not os.path.exists(f"{args.dir}/data_dir/train/"):
    os.makedirs(f"{args.dir}/data_dir/train/")
    os.makedirs(f"{args.dir}/data_dir/test/")
    os.makedirs(f"{args.dir}/data_dir/val/")

with open(f"{args.dir}/data_dir/train/{task_idx_real}.train.json", 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/data_dir/val/{task_idx_real}.val.json", 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/data_dir/val/{task_idx_real}.val.mini.json", 'w') as f:
    import random
    random.seed(1)
    random.shuffle(val_data)
    val_data = val_data[:10000]
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{args.dir}/data_dir/test/{task_idx_real}.test.json", 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n") 


task_real_ids = list(train_data_by_task.keys())
assert len(train_data_by_task) == len(val_data_by_task) == len(test_data_by_task)
for trid in task_real_ids:
    all_data = {"train": train_data_by_task,
                "val": val_data_by_task,
                "test": test_data_by_task}
    for split in all_data:
        data = all_data[split][trid]
        with open(f"{args.dir}/data_dir/{trid}.{split}.json", 'w') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        if split == "val":
            with open(f"{args.dir}/data_dir/{trid}.{split}.mini.json", 'w') as f:
                for item in data[:2000]:
                    f.write(json.dumps(item) + "\n")
