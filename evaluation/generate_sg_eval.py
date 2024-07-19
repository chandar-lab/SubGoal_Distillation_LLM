from scienceworld import ScienceWorldEnv
import argparse
import openai
import pandas as pd
import re
import os
import json
from retry.api import retry_call
from datasets import load_dataset

desc = 'You are in a simulated environment as an agent. ' \
       'A task and its description will be given to you. ' \
       'Suggest the best actions the agent can take based ' \
       'on the things you see and the items in your inventory to complete the task. ' \
       'Only use valid actions and objects. ' \
       'If you know what are around, then suggest ' \
       'the following actions. You are allowed to do the following actions with the objects. ' \
       'Open or close OBJ meaning open or close a container , Deactivate or activate OBJ meaning ' \
       'activate or deactivate a device, connect OBJ to OBJ meaning connect electrical components , ' \
       'disconnect OBJ meaning disconnect electrical components , use OBJ [on OBJ] meaning use a device/item ,' \
       ' look around meaning describe the current room, look at OBJ meaning describe an object in detail, ' \
       'look in OBJ meaning describe a container’s contents, read OBJ meaning read a note or book, ' \
       'move OBJ to OBJ meaning move an object to a container, pick up OBJ meaning move an object to the inventory,' \
       ' put down OBJ meaning drop an inventory item, pour OBJ into OBJ meaning pour a liquid into a container , ' \
       'dunk OBJ into OBJ meaning dunk a container into a liquid , mix OBJ meaning chemically mix a container , ' \
       'go to LOC meaning move to a new location , teleport to LOC meaning teleport to a specific room , ' \
       'eat OBJ meaning eat a food , flush OBJ meaning flush a toilet, focus on OBJ meaning signal intent ' \
       'on a task object, wait [DURATION] meaning take no action for some duration, task meaning describe ' \
       'current task, inventory meaning list agent’s inventory, OBJ means objects. LOC means location. ' \
       'There are 10 locations centered around a house theme. These are: kitchen, bathroom, workshop,  \
       art studio, greenhouse, outside, bedroom, living room, foundry.'


def sg_generator_eval(taskName, task_num, variation, task_desc, info_look, info_inventory):
    file_name = f"{'prompt_eval'}/{taskName}_{task_num}_gpt4"
    prompt_file = f"{'prompt_eval/eval.json'}"
    with open(prompt_file) as f:
        prompts = json.load(f)
    for item in prompts:
        if item['taskIdx'] == task_num:
            sample1 = item
    assert (sample1 == None, f"this task id is not defined in the prompts")
    messages = create_initialprompt_prompt(sample1)
    messages = create_initialprompt(task_desc, info_look, info_inventory, messages)

    sg_file_name = file_name + ".csv"
    if os.path.exists(sg_file_name):
        df = pd.read_csv(sg_file_name, encoding='latin-1')
        if df.loc[df.variations==variation, 'chatgpt_output'].any() :
            chatgpt_answer = df.loc[df.variations==variation, 'chatgpt_output'].values[0]
        else:
            chatgpt_answer = retry_call(apichatgpt_givenprompt, fargs=[messages], fkwargs=None, tries=10, delay=1,
                                                max_delay=10, backoff=1.2)
            recent_taskdecompition = pd.DataFrame({'variations': [variation], 'chatgpt_output': [chatgpt_answer]})
            recent_taskdecompition.to_csv(sg_file_name, mode='a', index=True, header=False)

    else:
        chatgpt_answer = retry_call(apichatgpt_givenprompt, fargs=[messages], fkwargs=None, tries=10, delay=1,
                                    max_delay=10, backoff=1.2)
        recent_taskdecompition = pd.DataFrame({'variations': [variation], 'chatgpt_output': [chatgpt_answer]})
        recent_taskdecompition.to_csv(sg_file_name, mode='a', index=True)

    subgoals, subgoals_list, actions_list = sg_extraction(chatgpt_answer)
    return subgoals, subgoals_list, actions_list


def create_initialprompt_prompt(sample1):
    ### for boiling
    task_desc1 = "Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
    lookaround1 = "This room is called the hallway. In it, you see: \n\tthe agent\n\ta substance called air\n\ta picture\nYou also see:\n\tA door to the art studio (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the greenhouse (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the living room (that is open)\n\tA door to the workshop (that is open)"
    inventory1 = "In your inventory, you see:\n\tan orange\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {"role": "user", "content": task_desc1 + lookaround1 + inventory1 +
                                    '. Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions.'}
    ]

    messages.append({"role": "assistant",
                     "content": "1- navigate_to(kitchen) : {'open door to kitchen', 'go to kitchen'} 2- pick_up(thermometer):{'pick up thermometer'} 3- find(metal pot):{'open cupboard', 'pick up metal pot'} 4- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 5- Focus_on(substance in metal pot):{'focus on substance in metal pot'} 6- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 7- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot'} 8- chek(steam): {'examine steam', 'use thermometer in inventory on steam'} 9- wait(2): {'wait1', 'wait1'}"})


    messages.append({"role": "user", "content": "New task Description: " + sample1['taskDescription'] +
                                                sample1['look_around'] + sample1['inventory'] +
                                                 ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions. "})

    messages.append({"role": "assistant",
                     "content": sample1['subgoal']})

    return messages

def create_initialprompt(task_desc, info_look, info_inventory, messages):

    messages.append({"role": "user", "content": "Based on this example complete the following. New task Description:"
                                                + task_desc + info_look + info_inventory +
                                                'Based on the given goal path, provide me with the functional format of high-level sub-tasks as the same as the example to complete this task and their correspondings actions.'})

    return messages

def apichatgpt_givenprompt(prompt):
    openai.api_key_path = 'api_key.txt' 

    completions = openai.ChatCompletion.create(
        model='gpt-4' , 
        n=1,
        messages=prompt,
        temperature=0.0,
    )

    instructions = completions.choices[0]['message']['content']

    return instructions

def sg_extraction(instr):
    subgoals_list = []
    actions_list = []
    subgoals = []

    for i in range(1, 40):
        if str(i) + '-' in instr:
            st = instr.find(str(i) + '-') + 3
            if str(i + 1) + '-' in instr:
                end = instr.find(str(i + 1) + '-')
                sg = instr[st:end]
                # print('sg with end is:', sg)
            else:
                sg = instr[st:]
                # print('sg without end is:', sg)
            if ':{' in sg:
                action = sg[sg.find(':{') + 2:-2].split(',')
                subgoal = sg[: sg.find(':{')]
            elif ': {' in sg:
                action = sg[sg.find(': {') + 3:-2].split(',')
                subgoal = sg[: sg.find(': {')]
            action = [re.sub('[{,}.]', '', act) for act in action]
            if '' in action:
                action.remove('')
            # add end subgoal flag to recognize a subgoal is finished
            action.append('end subgoal')
            # print('actions are:', action)
            subgoals_list.extend(subgoal for i in range(len(action)))
            actions_list.extend(action)
            ### add completed subgoals
            subgoals.append(subgoal)

    return subgoals, subgoals_list, actions_list


