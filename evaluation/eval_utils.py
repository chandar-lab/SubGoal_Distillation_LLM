
"""
The functions here are sourced from the repository at https://github.com/yuchenlin/swiftsage/

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoConfig
from math import ceil
import random
import re
from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data.data_utils import formalize_action, recover_action
import string 
import editdistance
import time 
import tiktoken 

action_type_description = [
    {"action_type": "WAIT()", "desc": "wait for something to be done, for example, an object on stove to be boiled"},
    {"action_type": "TELEPORT(room)", "desc": "directly go to a room such as TELEPORT(kitchen)"},
    # {"action_type": "LOOK(object)", "desc": "look at an object"},
    {"action_type": "READ(object)", "desc": "read an object such as a recipe or a book"},
    {"action_type": "PICK(object)", "desc": "pick up an object and put it into your inventory"},
    {"action_type": "OPEN(object)", "desc": "open an object with doors before you search or put things in it. For example, OPEN(freezer), OPEN(blast furnace)."},
    {"action_type": "ACTIVATE(object)", "desc": "activate and turn on an object such as sink or stove, so that you can use it. "},
    {"action_type": "DEACTIVATE(object)", "desc": "deactivate turn off the object"},
    {"action_type": "EXAMINE(object)", "desc": "look at an object carefully. For example, EXAMINE(apple). Note that you cannot EXAMINE a location."},
    {"action_type": "CONNECT(object)", "desc": "connect two objects so that they become useful"},
    {"action_type": "MOVE(object, place)", "desc": "move/place the object to a place"},
    {"action_type": "USE(object A, object B)", "desc": "use an object A on object B, for example, USE(thermometer in inventory, water) to check the temperature of water."},
    {"action_type": "MIX(container)", "desc": "mix the objects in a container such as MIX(cup containing sugar and water)"},
    {"action_type": "DUNK(object A, object B)", "desc": "dunk object A into object B (optional)"},
    {"action_type": "DROP(object A, object B)", "desc": "drop object A into object B (optional)"},
    {"action_type": "POUR(object A, object B)", "desc": "pour the object A into the container B; For example, POUR(red paint, glass cup)"},
    {"action_type": "FOCUS(object)", "desc": "focus on an important object that are required by the task description (e.g., a substance, a plant, an animal, and so on)."},
]

focus_on_count = {
    "0": 1, "1": 1, "2": 1, "3": 1, "4": 2, "5": 1, "6":1, "7":1,
    "8": 1, "9": 1, "10": 1, "11": 1, "12": 4, "13": 4, "14":1, "15":1,
    "16": 1, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22":1, "23":1,   
    "24": 1, "25": 1, "26": 2, "27": 1, "28": 1, "29": 2
    
}

rooms = ["hallway", "greenhouse", "green house", "kitchen", "bathroom", "outside", "workshop", "art studio", "foundry", "bedroom", "living room"]


def is_action_failed(obs):
    return obs == "No known action matches that input." or "can't" in obs or "not" in obs or "doesn't" in obs

def find_non_alpha_index(s):
    for i, c in enumerate(s):
        if not c.isalpha() and c != ' ':
            return i
    return -1  # if no non-alpha character found 

def clean_look(look, version="not_lite"):
    
    if "You also see:" in look:
        end_ind = look.index("You also see:")
        look = look[:end_ind]

    clean_looks = []
    for line in look.splitlines():
        if not line.strip():
            continue
        if "In it, you see:"  in line:
            if version != "lite":
                clean_looks.append(line)
            continue
        if "the agent" in line or " air" in line:
            continue
        line = line.replace("substance called ", " ").strip()
        if version == "lite":
            end_ind = find_non_alpha_index(line.strip())
            if end_ind > 0:
                line = line[:end_ind].strip()
        clean_looks.append(line)
    if version == "lite":
        return ", ".join(clean_looks)        
    else:
        return "\n \t - ".join(clean_looks[:])        


 
def get_current_room(look):
    global rooms 
    first_sent = look.split(".")[0]
    for r in rooms:
        if "called the "+ r in first_sent:
            return r  
    return None 


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args["lm_path"])
    if 'lama' in args["lm_path"]:
        lm_model = AutoModelForCausalLM.from_pretrained(args["lm_path"])
    else:
        lm_model = AutoModelForSeq2SeqLM.from_pretrained(args["lm_path"])
    lm_model.eval() 
    lm_model.to(device)
    if args["sbert"]:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        sbert_model = None 
    return lm_model, tokenizer, sbert_model

def load_model_sg(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args["sg_lm_path"])
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(args["sg_lm_path"])
    lm_model.eval()
    lm_model.to(device)
    if args["sbert"]:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        sbert_model = None
    return lm_model, tokenizer, sbert_model

def load_model_Firstsg(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args["Fsg_lm_path"])
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(args["Fsg_lm_path"])
    lm_model.eval()
    lm_model.to(device)
    if args["sbert"]:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        sbert_model = None
    return lm_model, tokenizer, sbert_model

def load_variation(env, args, task_num, logger):
    variations = []
    if (args["set"] == "train"):
        variations = list(env.getVariationsTrain())
        # train_len = min(50, len(variations))
        # variations = variations[:train_len]
        if task_num == 26:
            variations = variations[:int(len(variations)/10)]
        elif task_num == 29:
            variations = variations[:int(len(variations)/2)]
    elif (args["set"] == "test"):
        variations = list(env.getVariationsTest())
        if args["cut_off"]:
            test_len = min(50, len(variations))  #min(50, len(variations))
            random.seed(1)
            random.shuffle(variations)
            variations = variations[:test_len]
    elif (args["set"] == "dev"):
        variations = list(env.getVariationsDev()) 
        variations = variations[:3]
    elif (args["set"] == "test_mini_2"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[3:10] 
    elif (args["set"] == "test_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:3] 
    elif (args["set"] == "test_mini_mini"):
        variations = list(env.getVariationsTest()) 
        # random.seed(1)
        # random.shuffle(variations)
        variations = variations[:1] 
    else:
        logger.info("ERROR: Unknown set to evaluate on (" + str(args["set"]) + ")")
        exit(1)
 
    logger.info(variations)
    return variations




def findValidActionNew(predictions, env, look, recent_actions, sbert_model, logger, k=5, sg=False, taskName='', var=0, curr_subgoal='', task_description=''):
    if not sg:
        formal_predctions = predictions
        # predictions = [recover_action(x) for x in predictions]
        global rooms
        valid_open_door = ["open door to " + i for i in rooms] 
        invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
        validActions = set(env.getValidActionObjectCombinations())
        validActions.update(valid_open_door)
        validActions.difference_update(invalid_focus)

        inventory = env.inventory().lower()
        
        validActions.difference_update(recent_actions[-3:]) 

        if taskName=='chemistry-mix-paint-tertiary-color':
            predictions[0]=predictions[0].replace("paint in bowl", "paint in art studio in bowl")
            return predictions[0]
        if taskName=='chemistry-mix-paint-secondary-color':
            predictions[0]=predictions[0].replace("paint in jug", "paint in art studio in jug")
            return predictions[0]

        for va in list(validActions):
            if "door" in va and "open" not in va:
                validActions.remove(va)
                continue
            if va.startswith("focus on"): 
                pattern = re.compile(r"\b(?:focus|on|in|to)\b", re.IGNORECASE)
                used_objs = pattern.sub("", va).split(" ")
                valid = True
                for obj in used_objs:
                    if obj not in look + " " + inventory:
                        valid = False
                if not valid:
                    validActions.remove(va)
        

        # 1) if acton in top k is valid, choose it
        found_valid_in_top = False
        action = None
        num = 0 
        valid_num = 0
        for pred in predictions[:k]:
            pred = pred.replace("green house", "greenhouse") 
            num += 1
            if pred.strip() in validActions:
                found_valid_in_top = True
                action = pred.strip()
                valid_num = num
                break
        if found_valid_in_top:
            return action
        else:
            logger.info(f"No valid action found in top k={k} predictions.")
            validActions = list(validActions)
            validActions.sort(key=lambda x: len(x))
            logger.info("Valid Predictions: "+ str(validActions)) 
    

        # 2) else, find most similar action

        if sbert_model:    
            pred_vectors = sbert_model.encode(predictions[:5], batch_size=5, show_progress_bar=False)
            valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)


            # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
            similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

            # Take the sum of cosine similarities for each vector in valid_action_vectors
            sum_similarities = similarity_matrix.sum(axis=0)

            # Find the indices of the k vectors with the highest sum of cosine similarities
            N = 5 # Change this to the number of top vectors you want to retrieve
            top_indices = np.argpartition(sum_similarities, -N)[-N:]

            # Print the indices of the top vectors
            # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
            logger.info("The most similar valid actions to the predictions:")
            for ti in top_indices:
                logger.info("\t\t - "+validActions[ti])
            action = validActions[top_indices[0]]
        else:
            # jaccard
            topValue = 0.0
            topAction = predictions[0]
            valid_num = 0
            num = 0
            # embPred = sbert_model.encode(pred, convert_to_tensor=True)
            tokensPred = predictions[0].split(" ")
            uniqueTokensPred = set(tokensPred)
            
            for validAction in validActions: 
                tokensAction = validAction.split(" ")
                uniqueTokensAction = set(tokensAction)

                intersection = uniqueTokensPred.intersection(uniqueTokensAction)
                if (len(intersection) > topValue):
                    topAction = validAction
                    topValue = len(intersection)

            logger.info("TOP VALID ACTION: " + topAction)
            # Sanitize top action
            topAction = re.sub(r'[^A-Za-z0-9 ]+', '', topAction)
            action = topAction
    
    return action 
 

def getFilteredValidActions(env, look, filter=True, task_id=None, task_desc=None):
    global rooms
    valid_open_door = ["open door to " + i for i in rooms] 
    invalid_focus = ["focus on "+x for x in ["agent", "air"]+rooms]
    validActions = set(env.getValidActionObjectCombinations())
    validActions.update(valid_open_door)
    validActions.difference_update(invalid_focus)

    inventory = env.inventory()
    
    validActions.add("wait")
    validActions.add("wait1") 
    if task_id is not None and task_desc is not None: 
        if task_id not in [5,6,7,8,17,18,19,20]:
            for va in list(validActions):
                if not va.startswith("focus on"):
                    continue
                items = va.replace("focus on", "").split()
                task_desc = task_desc.translate(str.maketrans('', '', string.punctuation)).lower()
                if len(set(items) & set(task_desc.split())) == 0:
                    validActions.remove(va)
        if task_id not in [14,15,16]:
            for va in list(validActions):
                if not va.startswith("examine"):
                    continue
                items = va.replace("examine", "").split()
                task_desc = task_desc.translate(str.maketrans('', '', string.punctuation)).lower()
                if len(set(items) & set(task_desc.split())) == 0:
                    validActions.remove(va)
    for va in list(validActions):
        if not va.startswith("mix"):
            continue
        container_words = ["cup", "bowl", "metal pot", "jug"]
        if not any(["mix" + c for c in container_words]):
            validActions.remove(va)
    if not filter:
        return validActions
    for va in list(validActions):
        if "door" in va and "open" not in va:
            validActions.remove(va)
            continue
    return validActions
    
def sbert_search(action_list, validActions, sbert_model, logger, k=1, N=1, return_scores=False):
    validActions = list(validActions)
    pred_vectors = sbert_model.encode(action_list[:k], batch_size=5, show_progress_bar=False)
    valid_action_vectors = sbert_model.encode(validActions, batch_size=min(len(validActions), 128), show_progress_bar=False)

    # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
    similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

    # Take the sum of cosine similarities for each vector in valid_action_vectors
    sum_similarities = similarity_matrix.sum(axis=0)

    N = min(N, len(validActions))
    # Find the indices of the k vectors with the highest sum of cosine similarities
    # N = 10 # Change this to the number of top vectors you want to retrieve
    top_indices = np.argpartition(sum_similarities, -N)[-N:]

    # Print the indices of the top vectors
    # print(f"The indices of the top {k} vectors in valid_action_vectors are: {top_indices}")
    # logger.info("The most similar valid actions to the predictions:")
    # for ti in top_indices:
    #     logger.info("\t\t - "+validActions[ti])
    if N == 1:
        action = validActions[top_indices[0]]
        score = sum_similarities[top_indices[0]]
        if return_scores:
            return action, score
        return action
    else:
        action_list = []
        for i in range(N):
            action = validActions[top_indices[i]]
            action_list.append(action)
        return action_list



    

def find_object(action, objects_string): 
    # Find the index of the target object in the words list
    target_object = ' '.join(action.split()[2:])
    if target_object not in objects_string:
        return action 
    target_object_index = objects_string.index(target_object)
    
    # Check if the target object is inside a container
    if objects_string[target_object_index - 8:target_object_index - 1] == "called ":
        container_start_index = objects_string.rfind("(", 0, target_object_index) - 1
        container_end_index = objects_string.rfind(")", 0, target_object_index) + 1
        container = objects_string[container_start_index:container_end_index]
        action = action.replace(target_object, f"{container}")
    
    return action


def clean_obj_name(action):
    if "unknown substance" not in action:
        return action 
    for n in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        action = action.replace(f" {n}", "")
    return action 

def try_to_replace(action, validActions, look=None, inventory=None):
    if action.startswith("wait"):
        return "wait"
    if action in validActions:
        return action 
    try_action = action.replace("green house", "greenhouse") 
    try_action = try_action.replace("adult", "adult adult")
    try_action = try_action.replace("baby", "baby baby")
    if try_action in validActions:
        return try_action 
    if action.startswith("go to"):
        if action.replace("go to", "teleport to") in validActions:
            return action.replace("go to", "teleport to")
        elif action.replace("go to", "open door to") in validActions:
            return action.replace("go to", "open door to") 
    if action.startswith("pick up"):
        action = find_object(action, look)
        if action in validActions:
            return action 
        if action.replace("substance in ","") in validActions:
            return action
    if action.startswith("focus on"):
        obj = action.replace("focus on", "").strip()
        todo = "focus on substance in inventory"
        if obj in inventory and  todo in validActions:
            return todo
    if action.startswith("move") and "to" in action:
        pattern = r"move (.*?) to"
        obj = re.search(pattern, action)
        if obj is None:
            return action 
        else:
            obj = obj.group(1)
        todo = action.replace(obj, "substance in inventory")
        if obj in inventory and todo in validActions:
            return todo
    

    split_string = action.rsplit(" in ", 1) # Split the string from the last occurrence of " in "
    if split_string[0] in validActions:
        return split_string[0]

    if " unknown substance " in action:
        action = split_string[0]
        action = clean_obj_name(action)
        if action in validActions:
            return action 
        
    for r in rooms:
        action = action.replace("in " + r, "")
    return action 
        

def compose_prompt_to_nextactions(demos, task_desc, recent_actions, recent_obs, recent_locs, failed_messages, look, inventory, response_next_subgoal, useful_focus_on, fast_action=None, k=10, version="gpt-4"):

    prompt_to_next_actions = []
    prompt_to_next_actions.append("You are an experienced teacher who always guide students to complete the science experiments. Now let's do science experiments with a sequence of actions.")
    prompt_to_next_actions.append("In this environment, there are a few locations: art studio, workshop, kitchen, living room, bedroom, bathroom, foundry, greenhouse, outside, and a hallway connecting them.")
        
    prompt_to_next_actions.append("You have done a few science experiments successfully and below are the action history of your experiments with similar tasks.")

    prompt_to_next_actions.append("Example task 1: "+ demos[0][0])
    prompt_to_next_actions += demos[0][1:]
    if len(demos) >= 2:
        prompt_to_next_actions.append("Example task 2: "+ demos[1][0])
        prompt_to_next_actions += demos[1][1:]
    # prompt_to_next_actions += ["- Action: "+ a for a in demos[1][1:]]


    prompt_to_next_actions.append("In a new science experiment that is similar to the above two, " + task_desc.replace("Your", "my"))
    
    # prompt_to_next_actions.append("Given the above completed subgoals, what should be your next subgoal to complete for finishing the task?")
    
    prompt_to_next_actions.append(f"My previous {k} actions and observations are as follows:")

    recent_actions, recent_obs, _, _, recent_locs = clean_history(recent_actions, recent_obs, [-1]*len(recent_actions), [-1]*len(recent_actions), recent_locs)
        

    history = []
    repeat = 0    
    for ind, (l, a, o) in enumerate(zip(recent_locs[:], recent_actions[:], recent_obs[:])):
        if o == "N/A":
            continue 
        fa = formalize_action(a)
        if "(" not in fa:
            continue
        at = fa[:fa.index("(")]
        if at not in "\n".join(demos[0][1:]):
            # Skipping the actions with types not in the demos
            continue
        to_add = f"- (in {l}) Action: {fa} --> {o}"
        if ind+1 < len(recent_actions) and a in recent_actions[max(0, ind-5):ind] and a in recent_actions[ind+1:min(len(recent_actions), ind+5)]:
            repeat += 1
            continue 
        
        history.append(to_add) 
        if repeat > 0:
            history.append(f"Repeat the above action for {repeat} times.")             
            repeat = 0
    # prompt_to_next_actions.append()
    prompt_to_next_actions += history[-k:]

    if useful_focus_on:
        prompt_to_next_actions.append("Importantly, I have FOCUS on these things already: " + ", ".join([fo.replace("focus on", "") for fo in  useful_focus_on]))
    else:
        prompt_to_next_actions.append("Importantly, I have FOCUS on nothing yet.")

    pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc)
    to_focus = [match[0].replace("the ", " ").strip() for match in matches]
    pattern = r"find\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc.replace("a(n)", "a"))
    to_focus_v2 = [match[0].replace("the ", " ").strip() for match in matches]

    # prompt_to_next_actions.append("You have completed these subgoals:")
    # prompt_to_next_actions.append(response_previous_subgoals)
    prompt_to_next_actions.append("However, my actions so far cannot complete the task now. I do not know what to do for the next steps.")
    if failed_messages:
        failed_messages = set(failed_messages)
        prompt_to_next_actions.append("There are some error messages about my previous actions:")
        prompt_to_next_actions += failed_messages
    prompt_to_next_actions.append("I asked my teacher for advice and the teacher told me these advice:")
    prompt_to_next_actions.append(response_next_subgoal.replace("Question", "Answer").replace("Answer", "Advice")) 
    prompt_to_next_actions.append("")
    prompt_to_next_actions.append("In current environment: " + clean_look(look) + "\n" + inventory)
    prompt_to_next_actions.append("What should be my next actions to complete the next subgoal in the current environment? ")
    prompt_to_next_actions.append("If any of the suggested next subgoals need knowledge to make decisions (e.g., determining or comparing the properties of objects and animals), please do that for me.")
    prompt_to_next_actions.append("The ONLY allowed action types are:")
    for ai in action_type_description:
        at = ai['action_type']
        at = at[:at.index("(")]
        if at not in "\n".join(demos[0][1:] + demos[0][2:]):
            continue
        prompt_to_next_actions.append(f"- {ai['action_type']} : {ai['desc']} ")   

    prompt_to_next_actions.append(f"Important! You can only use FOCUS actions on these items: {', '.join(to_focus)} . ") # (Hint: {','.join(to_focus_v2)})
    prompt_to_next_actions.append("You cannot FOCUS on any other things. Please only use FOCUS as required by the task description. Also, please FOCUS more directly, try not to focus on the container.")

    prompt_to_next_actions.append("Please use the above mentioned action types to convert the unfinished subgoal to a short sequence of concrete actions.  DO NOT USER OTHER TYPES OF ACTIONS. Follow the report of the two example tasks shown to you previously.")    
    prompt_to_next_actions.append("Please do not try to look for books or computers to look up information. You will need to use your own commonsense knowledge to make decisions (e.g., determining properties of objects and animals).")
    prompt_to_next_actions.append("Note that I can only do actions with available objects in the current location or inventory!!") 
    prompt_to_next_actions.append("Please use the below format to organize the response.")
    prompt_to_next_actions.append("Action 1: [...] -->  \n Action 2: [...] --> \n ...")
    return "\n".join(prompt_to_next_actions)

 
def compose_prompt_to_plan(demos, useful_focus_on, task_desc, recent_actions, recent_obs, recent_locs, recent_looks, failed_messages, look, inventory, fast_action, version="full"):
    clean_obs = []
    assert len(recent_obs) == len(recent_locs)
    repeat = 0 
    for i, obs in enumerate(recent_obs[1:]):
        # if obs.startswith("This room is called"):
        #     end_index = obs.index("In it")
        #     obs = obs[:end_index]
        if obs.startswith("You move to the") or obs.startswith("You go to the") or obs.startswith("You teleport to the"):
            obs = obs.replace("go to", "move to").replace("teleport to", "move to")
        if obs == "The door is already open.":
            continue
        # if obs.startswith("a substance called"): 
        if f"In {recent_locs[i+1]}, {obs}" in clean_obs:
            continue
        
        if recent_actions[i+1] in recent_actions[i+1-5:i+1] and recent_actions[i+1] in recent_actions[i+2:i+2+5]:
            repeat += 1
            continue
        if "move to the" in obs:
            clean_obs.append(f"{obs}")
        else:
            if version == "lite":
                clean_obs.append(f"In {recent_locs[i+1]}, {obs}")
            else:
                clean_obs.append(f"In {recent_locs[i+1]}, {recent_actions[i+1]} --> {obs}")

        if repeat > 0: 
            clean_obs.append(f"Repeat the above {repeat} times.")        
            repeat = 0
    final_obs = []
    for i, co in enumerate(clean_obs):
        if i+1 < len(clean_obs) and "move to the" in clean_obs[i] and "move to the" in clean_obs[i+1]:
            continue
        final_obs.append(co.replace("a substance called", "there is a"))
    prev_obs = [f"- {j+1}. {o}" for j, o in enumerate(final_obs)]

    
    prompt_to_plan  = []

    prompt_to_plan.append("You are an experienced teacher who always guides students to complete the science experiments by giving executable advice and instructions with world knowledge.")

    prompt_to_plan.append("You have done a science experiment successfully and below is the action history of your experiment.")

    prompt_to_plan.append("Example task: "+ demos[0][0])
    clean_actions = []
    for history in demos[0][1:]:
        if "Action: " not in history:
            continue
        start_ind = history.index("Action: ") + len("Action: ")
        end_ind = history.index(" -->")
        action = history[start_ind:end_ind]
        action = recover_action(action)
        if action is not None:
            clean_actions.append(history[:start_ind] + action + history[end_ind:])
    prompt_to_plan += clean_actions

    prompt_to_plan.append("In a new science experiment that is similar to the above one, " + task_desc.replace("Your", "my")) 
    prompt_to_plan.append("In this environment, there are a few rooms: art studio, workshop, kitchen, living room, bedroom, bathroom, foundry, greenhouse, outside, and a hallway connecting them.")
    prompt_to_plan.append("To complete this task, I have done some actions and the observations are listed here:")
    if version == "lite":
        prev_obs = prev_obs[-15:]
    prompt_to_plan += prev_obs
    # print(recent_looks)
    # print(recent_locs)
    if len(recent_looks) >= 2 and version != "lite":
        prompt_to_plan.append("In some previously visited locations:")    
        for location, look_round in recent_looks.items():
            if location != recent_locs[-1]:
                prompt_to_plan.append(f"In {location}: " + clean_look(look_round, version="lite"))
    prompt_to_plan.append("* Current location *: " + clean_look(look)) # + look.replace(" egg", " ").replace(" adult ", " ").replace(" baby ", " ")
    prompt_to_plan.append(inventory.replace("Your ", "My "))
    if useful_focus_on:
        prompt_to_plan.append("Importantly, I have FOCUS on these things already: " + ", ".join([fo.replace("focus on", "") for fo in  useful_focus_on]))
    else:
        prompt_to_plan.append("Importantly, I have FOCUS on nothing yet.")
    # prompt_to_plan.append("However, my actions so far cannot complete the task. I do not know what to do for the next steps.")
    prompt_to_plan.append("However, I do not know what to do for the next steps.")
    if fast_action:
        prompt_to_plan.append(f"My instinct tells me that it might be reasonable to {fast_action} now but I'm not so sure.")
    if failed_messages:
        failed_messages = set(failed_messages)
        failed_messages = set(failed_messages)
        prompt_to_plan.append("There are some error messages about my previous actions:")
        prompt_to_plan += failed_messages
    prompt_to_plan.append("Please review the task description and the previous observations and then answer the following questions to help me plan for efficiently completing the next subgoal.")
    prompt_to_plan.append("Question 1: To efficiently complete the task, what substance and objects do I need to collect? Please list them and their possible locations one by one. Please ignore protective gears because I have them already.")
    prompt_to_plan.append("Question 2: Based on your answer to Question 1, are there any substance or objects that are not in my inventory now and I should keep looking for?" + \
                          " If so, which rooms are they likely to be? " + \
                          "Note that some of your suggested items might not exist in the rooms. In that case, let's try to use the similar ones in the environment." + \
                          " Note that I cannot do actions without them if they are not collected yet. ")
   

    pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, task_desc)
    to_focus = [match[0].replace("the ", " ").strip() for match in matches]

    prompt_to_plan.append("Question 3: To most efficiently complete the task, what will be the important subgoals to finish? Please list up to five subgoals." + \
                          f" Importantly, please include the subgoals about 'focus on' as required in the task description. Remember that it is ONLY possible focus on these items: {', '.join(to_focus)}! You should NOT focus on other things!! If you list a subgoal of focusing on, make sure that is mentioned and required by the task.")
    prompt_to_plan.append("Question 4: In these subgoals, what have I already completed based on the previous observations? And which subgoals should I aim to do right now?" + \
                          " These subgoals may need additional common knowledge to make decisions. Please recall the knowledge about the properties of objects or animals. Think step by step, and list the facts that are useful. And then use them for determining or comparing if needed. Finally, list the next subgoals based on the knowledge and current observations.")
    prompt_to_plan.append("Question 5: Based on the observations, did I make any mistakes that prevent me from efficiently finishing the next subgoals? Did I forget to go to a location to pick up thing? Or did I forget to open/activate/move something? Did I repeat any actions too many times? If so, how should I fix it?")
    prompt_to_plan.append("Please do not try to look for books or computers to look up information. You will need to use your own commonsense knowledge to make decisions (e.g., determining properties of objects and animals).")
    prompt_to_plan.append("Please read the task description carefully, and think step by step to answer these questions one by one. Please be concise. Thank you very much.")
    return '\n'.join(prompt_to_plan)

def clean_history(recent_actions, recent_obs, recent_score, recent_reward, recent_locs):
    assert len(recent_actions) == len(recent_obs) == len(recent_score) == len(recent_reward) == len(recent_locs)
    N = len(recent_actions)
    inds_to_remove = []
    for ind in range(N):
        if recent_actions[ind].startswith("examine"):
            inds_to_remove.append(ind)
        if recent_actions[ind].startswith("teleport to") and recent_score[ind] >= 0:
            recent_actions[ind] = recent_actions[ind].replace("teleport", "go")
            recent_obs[ind] = recent_obs[ind].replace("teleport", "go")
        if recent_actions[ind].startswith("go to") and recent_score[ind] < 0:
            recent_actions[ind] = recent_actions[ind].replace("go", "teleport")
            recent_obs[ind] = recent_obs[ind].replace("go", "teleport")
        if recent_actions[ind].startswith("open door") and recent_score[ind] < 0:
            inds_to_remove.append(ind)
        if recent_actions[ind] in recent_actions[ind+1: min(ind+3, N)] and recent_score[ind] >= 0 :
            inds_to_remove.append(ind)
    
    recent_actions = [item for idx, item in enumerate(recent_actions) if idx not in inds_to_remove]
    recent_obs = [item for idx, item in enumerate(recent_obs) if idx not in inds_to_remove]
    recent_score = [item for idx, item in enumerate(recent_score) if idx not in inds_to_remove]
    recent_reward = [item for idx, item in enumerate(recent_reward) if idx not in inds_to_remove]
    recent_locs = [item for idx, item in enumerate(recent_locs) if idx not in inds_to_remove]
    return recent_actions, recent_obs, recent_score, recent_reward, recent_locs

    
def get_model_output(args, input_str, tokenizer, lm_model, device, logger): 
    input_ids = tokenizer(input_str, return_tensors="pt", max_length=args["max_input_len"] , truncation=True).input_ids

    sample_outputs = lm_model.generate(
        input_ids.to(device),
        # max_length=30, #16
        num_return_sequences=args['beams'],
        num_beams=args['beams'], 
        temperature=0,
        # max_new_tokens=1024,
    )
 
    lm_pred = sample_outputs

    # Take the first prediction that is not "look around"
    logger.info("Top N Predictions:")
    predStrs = []
    for i, pred in enumerate(lm_pred):
        text = tokenizer.decode(pred)
        text = post_process_generation(text)
        logger.info("\t" + str(i) + "\t" + str(text) )
        predStrs.append(text)

    return predStrs


def post_process_generation(raw_pred):
    ans_match = re.match(r".*<extra_id_0>(.*)<extra_id_1>.*", raw_pred)
    if ans_match is not None:
        result = ans_match.group(1)
    else:
        result = raw_pred

    # remove extra <*>'s left in
    result = result.replace("<", " <")
    out = ""
    for token in result.split(" "):
        if (len(token.strip()) > 0):
            if (token[0] != "<"):
                out += token + " "
    result = out

    return result.strip()


def gpt_select_valid(action, candidates, look, inventory, goal, logger, n=1, gpt_version="gpt-4"):
    prompt_to_search = []
    prompt_to_search.append("Let's play a text game.")
    prompt_to_search.append(clean_look(look, version="all"))
    prompt_to_search.append(inventory)
    prompt_to_search.append("There are some action candidates as follows:")
    for ac in candidates:
        prompt_to_search.append(f"- {ac}")
    prompt_to_search.append(f"\n I want to achieve this goal: {goal} but my action '{action}' is not in the candidate list.")
    prompt_to_search.append(f"Please consider the objects in the room and inventory and my goal. Think carefully, and then select the best replacement from the list. If no one in the list is a good replacement, return 'none'.")
    prompt_to_search.append(f"Selected action:") 

    prompt_to_search = "\n".join(prompt_to_search)
    logger("-"*30 + "prompt_to_search" + "-"*30)
    logger("\n"+prompt_to_search)
    logger("-"*35 + "-"*35)

    responses = completion_with_backoff(model=gpt_version,
            messages=[{"role": "user", "content": prompt_to_search},  
                        ], n = n, temperature=0, top_p=1)
    # logger(responses)
    selections = [responses["choices"][i]["message"]["content"] for i in range(n)]
    logger("\n" + "Responses: \n" + "\n".join(selections))

    return selections  




def rank_candidates_by_common_words(query, candidates):
    """
    Rank the candidates based on their edit distance to the query.
    """

    # the first word must be the same 
    candidates = [va for va in candidates if va.split()[0] == query.split()[0]]
    
    # Compute the edit distance between each candidate and the query
    num_commons = [len(set(query.split()) & set(candidate.split())) for candidate in candidates]
    
    # Sort the candidates based on their distance to the query
    ranked_candidates = [candidate for _, candidate in sorted(zip(num_commons, candidates), reverse=True)]
    
    return ranked_candidates

if __name__ == "__main__":  
    print()
