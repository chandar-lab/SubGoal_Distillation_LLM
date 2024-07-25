from scienceworld import ScienceWorldEnv
import argparse
import openai
import pandas as pd
import re
import os
import json
from retry.api import retry_call
from datasets import load_dataset
import sys
from dynamic_prog_check import find_mismatched
import time
from ast import literal_eval

desc = (
    "You are in a simulated environment as an agent. "
    "A task and its description will be given to you. "
    "Suggest the best actions the agent can take based "
    "on the things you see and the items in your inventory to complete the task. "
    "Only use valid actions and objects. "
    "If you know what are around, then suggest "
    "the following actions. You are allowed to do the following actions with the objects. "
    "Open or close OBJ meaning open or close a container , Deactivate or activate OBJ meaning "
    "activate or deactivate a device, connect OBJ to OBJ meaning connect electrical components , "
    "disconnect OBJ meaning disconnect electrical components , use OBJ [on OBJ] meaning use a device/item ,"
    " look around meaning describe the current room, look at OBJ meaning describe an object in detail, "
    "look in OBJ meaning describe a container’s contents, read OBJ meaning read a note or book, "
    "move OBJ to OBJ meaning move an object to a container, pick up OBJ meaning move an object to the inventory,"
    " put down OBJ meaning drop an inventory item, pour OBJ into OBJ meaning pour a liquid into a container , "
    "dunk OBJ into OBJ meaning dunk a container into a liquid , mix OBJ meaning chemically mix a container , "
    "go to LOC meaning move to a new location , teleport to LOC meaning teleport to a specific room , "
    "eat OBJ meaning eat a food , flush OBJ meaning flush a toilet, focus on OBJ meaning signal intent "
    "on a task object, wait [DURATION] meaning take no action for some duration, task meaning describe "
    "current task, inventory meaning list agent’s inventory, OBJ means objects. LOC means location. "
    "There are 10 locations centered around a house theme. These are: kitchen, bathroom, workshop,  \
                     art studio, greenhouse, outside, bedroom, living room, foundry."
)

path_api_key = "api_key.txt"  # pass your API key here
mode_gpt = "gpt4"  # you can chnage it to either 'chatgpt' or 'gpt4'
mode_gpt_first = "gpt-4"  # for chatgpt choose 'gpt-3.5-turbo-0613'


def sg_generator(gold_path, task_desc, VarId, steps, curr_task_seq, task_name_id):
    """
    This function generates the subgoals
    """
    main_file_name = "prompts"
    file_name = f"{main_file_name}/{task_name_id}_{mode_gpt}"
    prompt_file = file_name + ".jsonl"
    sg_file_name = file_name + ".csv"
    sg_file_name_2 = file_name + "_4_steps_.csv"
    find_answer = False

    if os.path.exists(sg_file_name_2):
        df = pd.read_csv(sg_file_name_2, encoding="latin-1")
        if df.loc[df.variations == VarId, "chatgpt_output"].any():
            chatgpt_answer = df.loc[df.variations == VarId, "chatgpt_output"].values[0]
            find_answer = True
            subgoals_list_insteps = literal_eval(
                df.loc[df.variations == VarId, "subgoal"].values[0]
            )
            steps = literal_eval(df.loc[df.variations == VarId, "steps"].values[0])
            return steps, subgoals_list_insteps
    sg_file_name_2_chatgpt = f"{main_file_name}/{task_name_id}_chatgpt_4_steps_.csv"
    if os.path.exists(sg_file_name_2_chatgpt):
        df = pd.read_csv(sg_file_name_2_chatgpt, encoding="latin-1")
        if df.loc[df.variations == VarId, "chatgpt_output"].any():
            chatgpt_answer = df.loc[df.variations == VarId, "chatgpt_output"].values[0]
            find_answer = True
            subgoals_list_insteps = literal_eval(
                df.loc[df.variations == VarId, "subgoal"].values[0]
            )
            steps = literal_eval(df.loc[df.variations == VarId, "steps"].values[0])
            return steps, subgoals_list_insteps

    if os.path.exists(sg_file_name) and not find_answer:
        df = pd.read_csv(sg_file_name, encoding="latin-1")
        if df.loc[df.variations == VarId, "chatgpt_output"].any():
            chatgpt_answer = df.loc[df.variations == VarId, "chatgpt_output"].values[0]
            find_answer = True

    if not find_answer:
        if os.path.exists(prompt_file):
            messages = [json.loads(line) for line in open(prompt_file, "r")]
        else:
            sample1, sample2 = create_samples(curr_task_seq)
            messages = create_initialprompt_start(
                gold_path, task_desc, sample1, sample2, task_name_id, prompt_file
            )
        messages = create_initialprompt(gold_path, task_desc, messages)
        chatgpt_answer = retry_call(
            apichatgpt_givenprompt,
            fargs=[messages, mode_gpt_first],
            fkwargs=None,
            tries=10,
            delay=1,
            max_delay=10,
            backoff=1.3,
        )

    actions_list, subgoal_action_index_list = sg_extraction_completion(chatgpt_answer)
    subgoal_action_index_list = check_action_sg(
        actions_list, gold_path, subgoal_action_index_list, task_desc, chatgpt_answer
    )
    steps, subgoals_list_insteps = align_sg_step(steps, subgoal_action_index_list)

    recent_taskdecompition = pd.DataFrame(
        {
            "variations": [VarId],
            "chatgpt_output": [chatgpt_answer],
            "subgoal": [str(subgoals_list_insteps)],
            "path_act_sg": [str(subgoal_action_index_list)],
            "steps": [str(steps)],
        }
    )

    if not os.path.exists(sg_file_name_2):
        recent_taskdecompition.to_csv(sg_file_name_2)
    else:
        recent_taskdecompition.to_csv(
            sg_file_name_2, mode="a", index=True, header=False
        )

    return steps, subgoals_list_insteps


def create_samples(curr_task_seq):
    """This function create samples for the prompting."""
    samples = []

    for seq_sample in curr_task_seq:
        task_desc = seq_sample["taskDescription"]
        VarId = seq_sample["variationIdx"]
        places = []
        objects = []

        original_steps = seq_sample["path"]

        steps = []
        for s in original_steps:
            if s["action"] == "look around":
                continue
            if s["action"].startswith("close door"):
                continue
            steps.append(s)

        if len(steps) < 2:
            continue

        fold = seq_sample["fold"]
        obs = steps[0]["observation"]
        action = steps[0]["action"]
        gold_length = len(steps)
        # filtering steps

        gold_path_only = []

        for item in steps:
            gold_path_only.append(item["action"])

        sample_ = {"task_desc": task_desc, "gold_path": gold_path_only}
        samples.append(sample_)

    return samples[0], samples[1]


def create_initialprompt_start(sample1, prompt_file):
    """This function create samples for the prompting using the samples."""
    task_desc1 = "Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
    path1 = [
        "open door to kitchen",
        "go to kitchen",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine steam",
        "use thermometer in inventory on steam",
        "wait1",
        "wait1",
    ]
    goalpath1 = ""
    for step in path1:
        goalpath1 += step + ", "

    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {
            "role": "user",
            "content": task_desc1
            + "Here is the goal path to achieve to the goal:"
            + goalpath1
            + ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions.",
        },
    ]

    messages.append(
        {
            "role": "assistant",
            "content": "1- navigate_to(kitchen) : {'open door to kitchen', 'go to kitchen'} 2- pick_up(thermometer):{'pick up thermometer'} 3- find(metal pot):{'open cupboard', 'pick up metal pot'} 4- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 5- Focus_on(substance in metal pot):{'focus on substance in metal pot'} 6- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 7- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot'} 8- chek(steam): {'examine steam', 'use thermometer in inventory on steam'} 9- wait(2): {'wait1', 'wait1'}",
        }
    )

    path_sample = sample1["gold_path"]
    goalpath_sample = ""
    for step in path_sample:
        goalpath_sample += step + ", "

    messages.append(
        {
            "role": "user",
            "content": "New task Description: "
            + sample1["task_desc"]
            + " Here is the goal path to achieve to the goal: "
            + goalpath_sample
            + ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions. ",
        }
    )

    ## Call GPT4 or ChatGPT
    GPT4output = apichatgpt_givenprompt(messages)
    messages.append({"role": "assistant", "content": GPT4output})

    with open(prompt_file, "w") as f:
        for item in messages:
            f.write(json.dumps(item) + "\n")

    return messages


def create_initialprompt(path2, task_desc2, messages):
    """This function create samples for the prompting for the second task"""
    goalpath2 = ""
    for step in path2:
        goalpath2 += step + ", "

    messages.append(
        {
            "role": "user",
            "content": "Based on this example complete the following. New task Description:"
            + task_desc2
            + "Here is  the goal path to achieve to the goal:"
            + goalpath2
            + "Based on the given goal path, provide me with the functional format of high-level sub-tasks as the same as the example to complete this task and their correspondings actions.",
        }
    )

    return messages


def create_initialprompt_react(path2, task_desc2):
    ### creating initial prompt for react
    task_desc1 = "Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter."
    path1 = [
        "open door to kitchen",
        "go to kitchen",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine steam",
        "use thermometer in inventory on steam",
        "wait1",
    ]
    goalpath1 = ""
    for step in path1:
        goalpath1 += step + ", "

    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {
            "role": "user",
            "content": task_desc1
            + "Here is the goal path to achieve to the goal:"
            + goalpath1
            + ". Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions.",
        },
    ]

    messages.append(
        {
            "role": "assistant",
            "content": "1- navigate_to(kitchen) : {'open door to kitchen', 'go to kitchen'} 2- pick_up(thermometer):{'pick up thermometer'} 3- find(metal pot):{'open cupboard', 'pick up metal pot'} 4- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 5- Focus_on(substance in metal pot):{'focus on substance in metal pot'} 6- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 7- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot'} 8- chek(steam): {'examine steam', 'use thermometer in inventory on steam', 'wait1'}",
        }
    )

    messages.append(
        {
            "role": "user",
            "content": "New task Description: 'Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.' Here is the goal path to achieve to the goal: 'open door to hallway', 'go to hallway', 'open door to kitchen', 'go to kitchen', 'pick up thermometer', 'open cupboard', 'pick up metal pot', 'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot', 'focus on substance in metal pot', 'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove', 'pick up metal pot', 'open door to outside', 'go to outside', 'open door to foundry', 'go to foundry', 'open blast furnace', 'move metal pot to blast furnace', 'activate blast furnace', 'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on steam', 'wait1'. Based on the given goal path, provide me with the functional format of high-level sub-tasks to complete this task and their correspondings actions. ",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": "1- navigate_to(hallway): {'open door to hallway', 'go to hallway'} 2- navigate_to(kitchen): {'open door to kitchen', 'go to kitchen'} 3- pick_up(thermometer): {'pick up thermometer'} 4- find(metal pot): {'open cupboard', 'pick up metal pot'} 5- Fill(metal pot, water): {'move metal pot to sink', 'activate sink', 'deactivate sink', 'pick up metal pot'} 6- Focus_on(substance in metal pot): {'focus on substance in metal pot'} 7- heat(water, metal pot): {'pour metal pot into metal pot', 'pick up metal pot', 'move metal pot to stove', 'activate stove'} 8- navigate_to(outside): {'open door to outside', 'go to outside'} 9- navigate_to(foundry): {'open door to foundry', 'go to foundry'} 10- heat(water, blast furnace): {'open blast furnace', move metal pot to blast furnace', 'activate blast furnace'} 11- Monitor_temperature(metal pot, boil): {'examine substance in metal pot', 'use thermometer in inventory on substance in metal pot', 'examine substance in metal pot', 'use thermometer in inventory on steam'} 12- wait(1): {'wait1'}",
        }
    )

    goalpath2 = ""
    for step in path2:
        goalpath2 += step + ", "

    messages.append(
        {
            "role": "user",
            "content": "Based on this example complete the following. New task Description:"
            + task_desc2
            + "Here is  the goal path to achieve to the goal:"
            + goalpath2
            + "Based on the given goal path, provide me with the functional format of high-level sub-tasks as the same as the example to complete this task and their correspondings actions.",
        }
    )

    return messages


def apichatgpt_givenprompt(prompt, lm="gpt-4"):
    """This function uses API instead of a language model.
    Here we use GPT3 API text-davinci-003 which is the most powerful model"""
    openai.api_key_path = path_api_key

    completions = openai.ChatCompletion.create(
        model=lm,
        n=1,
        messages=prompt,
        temperature=0.0,
    )

    instructions = completions.choices[0]["message"]["content"]

    return instructions


def sg_extraction(instr, steps):
    """This function extract subgoals from the generates text from API"""

    new_steps = []
    subgoals_list = []
    actions_list = []
    subgoals = []
    subgoals_list_insteps = []
    modified_answer = ""
    num_act = 0
    nums = [int(s) for s in re.findall(r"\b\d+\b", instr)]
    for i in range(1, 40):
        if str(i) + "-" in instr:
            st = instr.find(str(i) + "-") + 3
            if str(i + 1) + "-" in instr:
                end = instr.find(str(i + 1) + "-")
                sg = instr[st:end]
                print("sg with end is:", sg)
            else:
                sg = instr[st:]
                print("sg without end is:", sg)
            if ":{" in sg:
                action = sg[sg.find(":{") + 2 : -2].split(",")
                subgoal = sg[: sg.find(":{")]
            elif ": {" in sg:
                action = sg[sg.find(": {") + 3 : -2].split(",")
                subgoal = sg[: sg.find(": {")]
            action = [re.sub("[{,}.]", "", act) for act in action]
            if "" in action:
                action.remove("")
            subgoals_list.extend(subgoal for i in range(len(action)))
            actions_list.extend(action)
            ### add completed subgoals
            subgoals.append(subgoal)
            added_sg = False
            valid_actions = []

            ### align subgoals to actions in steps
            for act in action:
                act = act.replace("'", "")
                if act.strip() == steps[num_act]["action"]:
                    st = steps[num_act]
                    st["subgoal"] = subgoal
                    print(" current detected sg is :", subgoal)
                    valid_actions.append(act)
                    if not added_sg:
                        subgoals_list_insteps.append(subgoal)
                        print(" subgoals_list_insteps : ", subgoals_list_insteps)
                        added_sg = True
                        modified_answer += (
                            str(len(subgoals_list_insteps)) + "- " + subgoal + ": {"
                        )
                    modified_answer += act + ", "
                    new_steps.append(st)
                    num_act += 1
                    if num_act == len(steps):
                        break
                else:
                    continue
            if added_sg:
                modified_answer += "}\n"
                item = {
                    "action": "end subgoal",
                    "observation": "the current subgoal is done.",
                    "score": steps[num_act - 1]["score"],
                    "isCompleted": steps[num_act - 1]["isCompleted"],
                    "freelook": steps[num_act - 1]["freelook"],
                    "inventory": steps[num_act - 1]["inventory"],
                    "subgoal": subgoal,
                }
                new_steps.append(item)

    return new_steps, subgoals_list_insteps, modified_answer


def sg_extraction_completion(instr):
    subgoals_list = []
    actions_list = []  ## list of all actions
    subgoals = []
    num_action = 0
    subgoal_action_dict = {}
    subgoal_action_index_list = []

    for i in range(1, 40):
        if str(i) + "-" in instr:
            st = instr.find(str(i) + "-") + 3
            if str(i + 1) + "-" in instr:
                end = instr.find(str(i + 1) + "-")
                sg = instr[st:end]
            else:
                sg = instr[st:]
            if ":{" in sg:
                action = sg[sg.find(":{") + 2 : -2].split(",")
                subgoal = sg[: sg.find(":{")]
            elif ": {" in sg:
                action = sg[sg.find(": {") + 3 : -2].split(",")
                subgoal = sg[: sg.find(": {")]
            action = [re.sub("[{,}.]", "", act) for act in action]
            if "" in action:
                action.remove("")

            subgoals_list.extend(subgoal for i in range(len(action)))

            ### add completed subgoals
            subgoals.append(subgoal)
            subgoal_action_dict[subgoal] = action
            ### align subgoals to actions in steps
            for act in action:
                act = act.replace("'", "").strip()
                actions_list.append(act)
                num_action += 1
                subgoal_action_index_list.append(
                    {"action": act, "sg": subgoal, "idx": num_action - 1}
                )

    return actions_list, subgoal_action_index_list


def check_action_sg(
    actions_list, gold_path, sg_act_idx_list, task_desc, chatgpt_answer
):
    """This function check the matching action step with the generated actions"""
    print(f"len goal path {len(gold_path)} ---- len action seq {len(actions_list)}")
    x, y = find_mismatched(gold_path, actions_list)
    add_list = y[1]  # it is tuple
    remove_list = y[2]  # it is only a list
    actions_list[-1] = gold_path[-1]
    sg_act_idx_list[-1]["action"] = gold_path[-1]
    if len(add_list) == 0 and len(remove_list) == 0:
        return sg_act_idx_list
    # for adding subgoals
    ctr_add = 0
    j_idx = 0
    while ctr_add < len(add_list) - 1:
        check = False
        if add_list[ctr_add][0] + 1 == add_list[ctr_add + 1][0]:
            ctr_add += 1
            check = True
        if ctr_add == len(add_list) - 1:
            if add_list[ctr_add][0] - 1 == add_list[ctr_add - 1][0]:
                check = False
        if not check or ctr_add == len(add_list):

            if j_idx != ctr_add:

                print(" more than one action is left, there should be a sub-goal here.")
                missed_action_list = gold_path[
                    add_list[j_idx][0] : add_list[ctr_add][0] + 1
                ]
                ## call chatgpt to genrate a subgoal for missed parts
                messages = create_prompt_inmiddle(
                    missed_action_list, task_desc, gold_path, chatgpt_answer
                )
                chatgpt_answer_middle = retry_call(
                    apichatgpt_givenprompt,
                    fargs=[messages, "gpt-4"],
                    fkwargs=None,
                    tries=20,
                    delay=1,
                    max_delay=10,
                    backoff=1.5,
                )

                actions_list_middle, sg_act_idx_list_middle = sg_extraction_completion(
                    chatgpt_answer_middle
                )
                index = add_list[j_idx][1]

                find_idx = False
                for i in range(len(sg_act_idx_list)):
                    if sg_act_idx_list[i]["idx"] == index:
                        for j in range(len(sg_act_idx_list_middle)):
                            item = {
                                "action": sg_act_idx_list_middle[j]["action"],
                                "sg": sg_act_idx_list_middle[j]["sg"],
                                "idx": "*",
                            }
                            sg_act_idx_list.insert(i + j, item)
                        find_idx = True
                        break

                if find_idx == False:
                    assert "the index is not find in the action list"

            else:
                missed_action = gold_path[add_list[j_idx][0]]
                index = add_list[j_idx][1]
                for i in range(len(sg_act_idx_list)):
                    if sg_act_idx_list[i]["idx"] == index:
                        if i > 0:
                            item = {
                                "action": missed_action,
                                "sg": sg_act_idx_list[i - 1]["sg"],
                                "idx": "*",
                            }
                        else:
                            item = {
                                "action": missed_action,
                                "sg": sg_act_idx_list[i]["sg"],
                                "idx": "*",
                            }
                        sg_act_idx_list.insert(i, item)

            ctr_add += 1
            j_idx = ctr_add

    if len(add_list) == 1:
        j_idx = 0
        missed_action = gold_path[add_list[j_idx][0]]
        index = add_list[j_idx][1]
        for i in range(len(sg_act_idx_list)):
            if sg_act_idx_list[i]["idx"] == index:
                if i > 0:
                    item = {
                        "action": missed_action,
                        "sg": sg_act_idx_list[i - 1]["sg"],
                        "idx": "*",
                    }
                else:
                    item = {
                        "action": missed_action,
                        "sg": sg_act_idx_list[i]["sg"],
                        "idx": "*",
                    }
                sg_act_idx_list.insert(i, item)

    # for removing extra actions
    ctr1 = 0
    i_idx = 0
    while ctr1 < len(remove_list):
        if sg_act_idx_list[i_idx]["idx"] == remove_list[ctr1]:
            sg_act_idx_list.pop(i_idx)
            ctr1 += 1
        else:
            i_idx += 1

    new_action_l = []
    for total_idx in range(len(sg_act_idx_list)):
        sg_act_idx_list[total_idx]["idx"] = total_idx
        new_action_l.append(sg_act_idx_list[total_idx]["action"])

    return check_action_sg(
        new_action_l, gold_path, sg_act_idx_list, task_desc, chatgpt_answer
    )


def create_prompt_inmiddle(missed_action_list, task_desc, gold_path, chatgpt_answer):
    """This function create prompting for the missed action."""
    goalpath1 = ""
    for step in gold_path:
        goalpath1 += step + ", "

    messages = [
        {"role": "system", "content": "You are a helpful assistant." + desc},
        {
            "role": "user",
            "content": task_desc
            + "Here is the goal path to achieve to the goal:"
            + goalpath1
            + ". Based on the given goal path, you already provided me with the functional format of high-level sub-tasks to complete this task and their correspondings actions."
            "which are: " + chatgpt_answer,
        },
    ]

    messages.append(
        {
            "role": "user",
            "content": ". Now, here is the goal path"
            + str(missed_action_list)
            + " What is the functional format of high-level sub-task assigned to these actions? Tell them in numeric format.",
        }
    )

    return messages


def align_sg_step(steps, sg_act_idx_list):
    steps_cp = steps.copy()
    if len(steps_cp) != len(sg_act_idx_list):
        print(" something is wrong ")

    subgoals_list_insteps = []
    i = 0
    sg_cnt = 0
    while i < (len(steps_cp)):

        isg = i - sg_cnt
        if sg_act_idx_list[isg]["action"] == steps_cp[i]["action"]:
            steps_cp[i]["subgoal"] = sg_act_idx_list[isg]["sg"]
        else:
            print(" .... actions are still not aligned ....")
            print(
                f"index is {i} , steps action is :",
                steps_cp[i]["action"],
                "---- gpt action is :",
                sg_act_idx_list[isg]["action"],
            )
        if i + 1 - sg_cnt < len(sg_act_idx_list):
            if sg_act_idx_list[isg + 1]["sg"] != sg_act_idx_list[isg]["sg"]:
                item = {
                    "action": "end subgoal",
                    "observation": "the current subgoal is done.",
                    "score": steps_cp[i]["score"],
                    "isCompleted": steps_cp[i]["isCompleted"],
                    "freelook": steps_cp[i]["freelook"],
                    "inventory": steps_cp[i]["inventory"],
                    "subgoal": sg_act_idx_list[isg]["sg"],
                }
                steps_cp.insert(i + 1, item)
                subgoals_list_insteps.append(sg_act_idx_list[isg]["sg"])
                i += 1
                sg_cnt += 1
        i += 1

    subgoals_list_insteps.append(sg_act_idx_list[-1]["sg"])

    return steps_cp, subgoals_list_insteps
