
import re
import random 


def add_current_place(curr_obs, look, places):
    
    # Extract current place
    index = curr_obs.find("move to the")
    if index != -1:
        place = curr_obs[index + 12: -1].strip()
        if place not in places:
            places.append(place)
    
    # Extract the first room the where the agent is 
    start = look.find("This room is called the")
    end = look.find(".")    
    if start != -1:
        if look[start + 24 : end] not in places:
            places.append(look[start + 24 : end])
    return


def get_real_task_id(task_name):
    task_name = task_name.replace("mendelian", "mendellian")
    task_table = {
        'boil': '1-1',
        'melt': '1-2',
        'freeze': '1-3',
        'change-the-state-of-matter-of': '1-4',
        'use-thermometer': '2-1',
        'measure-melting-point-known-substance': '2-2',
        'measure-melting-point-unknown-substance': '2-3',
        'power-component': '3-1',
        'power-component-renewable-vs-nonrenewable-energy': '3-2',
        'test-conductivity': '3-3',
        'test-conductivity-of-unknown-substances': '3-4',
        'find-living-thing': '4-1',
        'find-non-living-thing': '4-2',
        'find-plant': '4-3',
        'find-animal': '4-4',
        'grow-plant': '5-1',
        'grow-fruit': '5-2',
        'chemistry-mix': '6-1',
        'chemistry-mix-paint-secondary-color': '6-2',
        'chemistry-mix-paint-tertiary-color': '6-3',
        'lifespan-longest-lived': '7-1',
        'lifespan-shortest-lived': '7-2',
        'lifespan-longest-lived-then-shortest-lived': '7-3',
        'identify-life-stages-1': '8-1',
        'identify-life-stages-2': '8-2',
        'inclined-plane-determine-angle': '9-1',
        'inclined-plane-friction-named-surfaces': '9-2',
        'inclined-plane-friction-unnamed-surfaces': '9-3',
        'mendellian-genetics-known-plant': '10-1',
        'mendellian-genetics-unknown-plant': '10-2'
    }
    return task_table.get(task_name)


def add_current_objects(task_id, look, objects, limit=20):
    if True:
        things = re.findall(r'a .*?\n', look.replace(',', '\n').replace(
            '.', '\n').replace('(', '\n').replace(')', ' ').replace(' an ', ' a '), re.I)
    else:
        things = re.findall(r'a .*?\n', look.replace(',',
                            '\n').replace('.', '\n').replace(' an ', ' a '), re.I)

    flag = 0
    for thing in things:
        if 'door' in thing:
            continue

        if ('(' in thing) and (')' not in thing):
            start = things.index(thing)
            flag = 1

        if (')' in thing) and ('(' not in thing):
            end = things.index(thing)
            thing = 'and '.join(things[start: end + 1])
            flag = 0

        if flag:
            continue

        if clean(thing).strip().strip('.') not in objects:
            objects.append(clean(thing).strip().strip('.'))

    while len(objects) > limit:  
        objects.pop(0)
    return


def sanitizeStr(inStr):
    if inStr is None:
        return inStr
    out = inStr.replace("\n\t", " | ").replace("\n", " | ").replace("\t", " | ") 
    out = out.replace("green house", "greenhouse")
    out = out.replace("activate bee hive", "open bee hive")
    out = out.replace("seed peach tree", "peach seed")
    out = out.replace("orange tree seed", "orange seed")
    return out


def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s


def downsampling(task_idx_real, curr_task_seq):
    # Downsampling Task 26 and 29
    if task_idx_real.startswith('9-'):
        random.seed(1)
        random.shuffle(curr_task_seq)
        curr_task_seq = curr_task_seq[:50]
    elif task_idx_real.startswith('10-'):
        random.seed(1)
        random.shuffle(curr_task_seq)
        curr_task_seq = curr_task_seq[:50]
    elif task_idx_real.startswith('3-3'):
        random.seed(1)
        random.shuffle(curr_task_seq)
        curr_task_seq = curr_task_seq[:100]
    return curr_task_seq


def action_conversion(action, pattern, format_str, num_args):
        if num_args == 0:
            if action.strip() == pattern.strip():
                return format_str
            else:
                return None
        match = re.search(pattern, action)
        if match:
            if num_args == 1:
                formatted_action = format_str.format(match.group(1))
            elif num_args == 2:
                formatted_action = format_str.format(match.group(1), match.group(2))
            return formatted_action
        return None 


def compose_instance_v4_sg(step_id, task_desc, returns_to_go, curr_action,
                     curr_obs, inventory, look, prev_action, prev_obs, 
                     objects, places, recent_actions, recent_obs, recent_scores, recent_reward, subgoals):

    ''' In this function the labels/predictions are the actions with storing subgoals and it keeps the history of (action, observation)..'''

    label = curr_action
        
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1]*100)}; </s> "

    if len(subgoals)> 1:
        input_str += '. Completed subtasks are: ' + ', '.join(sg for sg in subgoals[:-1])
    else:
        ### the completed subgoals are empty
        input_str += '. No completed subtasks'

    input_str += '. The current subtask is ' + subgoals[-1]    
    
    input_str += "Action history: </s>" 
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward*100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> " 

    input_str += "Current environment: " + look + " </s> " 
    input_str += "Current inventory: " + inventory + " </s> "
    
    if places:
        input_str +=  "Visited rooms: " + ", ".join(places)  + ' </s> '
        
    input_str += ' What action should you do next? </s> '
    
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "") 
    if label != None:
        action_formatted = formalize_action(sanitizeStr(label))
        if action_formatted == None:
            print(label)
            raise Exception
    else:
        action_formatted = None 
    return input_str, action_formatted


def compose_instance_v4_sgaslabel(step_id, task_desc, returns_to_go, curr_action,
                           curr_obs, inventory, look, prev_action, prev_obs,
                           objects, places, recent_actions, recent_obs, recent_scores, recent_reward, subgoals, curr_subgoal):

    ''' In this function the labels/predictions are the subgoals instead of the actions and it keeps the history of (action, observation)..'''

    label = curr_subgoal

    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "

    if len(subgoals) > 2:
        input_str += '. Completed subtasks are: ' + ', '.join(sg for sg in subgoals[:-2])
    else:
        ### the completed subgoals are empty
        input_str += '. No completed subtasks'

    if len(subgoals) > 1:
        input_str += '. The current subtask is ' + subgoals[-2]

    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' . What subtask should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")

    return input_str, label


def compose_instance_v4_sgaslabel_test(step_id, task_desc, returns_to_go, curr_action,
                           curr_obs, inventory, look, prev_action, prev_obs,
                           objects, places, recent_actions, recent_obs, recent_scores, recent_reward, subgoals, curr_subgoal):

    ''' In this function the labels/predictions are the subgoals instead of the actions and it keeps the history of (action, observation)..'''

    label = curr_subgoal

    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "

    if len(subgoals) > 1:
        input_str += '. Completed subtasks are: ' + ', '.join(sg for sg in subgoals[:-1])
    else:
        ### the completed subgoals are empty
        input_str += '. No completed subtasks'

    if len(subgoals) > 0:
        input_str += '. The current subtask is ' + subgoals[-1]

    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' . What subtask should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")

    return input_str, label


def compose_instance_v4_sgaslabel_v4(step_id, task_desc, returns_to_go, curr_action,
                           curr_obs, inventory, look, prev_action, prev_obs,
                           objects, places, recent_actions, recent_obs, recent_scores, recent_reward, subgoals, curr_subgoal):

    ''' In this function the labels/predictions are the subgoals instead of the actions and it keeps the history of (action, observation)..'''

    label = curr_subgoal

    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "

    if len(subgoals) > 1:
        input_str += '. Completed subtasks are: ' + ', '.join(sg for sg in subgoals[:-2])
    else:
        ### the completed subgoals are empty
        input_str += '. No completed subtasks'

    input_str += '. The current subtask is ' + subgoals[-2]

    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' . What action should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")

    return input_str, label


def compose_instance_v4_sg_nohist(step_id, task_desc, returns_to_go, curr_action,
                           curr_obs, inventory, look, prev_action, prev_obs,
                           objects, places, recent_actions, recent_obs, recent_scores, recent_reward, subgoals):
    
    ''' In this function the labels/predictions are the actions and it has subgoals but it does not keep the history of (action, observation).'''
    
    label = curr_action

    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "

    if len(subgoals) > 1:
        input_str += '. Completed subtasks are: ' + ', '.join(sg for sg in subgoals[:-1])
    else:
        ### the completed subgoals are empty
        input_str += '. No completed subtasks'

    input_str += '. The current subtask is ' + subgoals[-1]

    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-1:], recent_actions[-1:], recent_reward[-1:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' What action should you do next? </s> '

    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")
    if label != None:
        action_formatted = formalize_action(sanitizeStr(label))
        if action_formatted == None:
            print(label)
            raise Exception
    else:
        action_formatted = None
    return input_str, action_formatted


def compose_instance_v5(step_id, task_desc, returns_to_go, curr_action,
                     curr_obs, inventory, look, prev_action, prev_obs, 
                     objects, places, recent_actions, recent_obs, recent_scores, recent_reward):

    ''' In this function the labels/predictions are the actions without storing subgoals and it keeps the history of (action, observation).'''
    
    label = curr_action
        
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1]*100)}; </s> "
    
    input_str += "Action history: </s>" 
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {formalize_action(action)} (+{int(reward*100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> " 

    input_str += "Current environment: " + look + " </s> " 
    input_str += "Current inventory: " + inventory + " </s> "
    
    if places:
        input_str +=  "Visited rooms: " + ", ".join(places)  + ' </s> '

        
    input_str += ' What action should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "") 
    if label != None:
        action_formatted = formalize_action(sanitizeStr(label))
        if action_formatted == None:
            print(label)
            raise Exception
    else:
        action_formatted = None 
    return input_str, action_formatted


def compose_instance_v4(step_id, task_desc, returns_to_go, curr_action,
                     curr_obs, inventory, look, prev_action, prev_obs, 
                     objects, places, recent_actions, recent_obs, recent_scores, recent_reward):

    ''' In this function the labels/predictions are the actions without storing subgoals and it keeps the history of (action, observation).'''

    label = curr_action
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1]*100)}; </s> "
    input_str += "Action history: </s>" 
    ind = 10
    for obs, action, reward in zip(recent_obs[-10:], recent_actions[-10:], recent_reward[-10:]):
        input_str += f" <extra_id_{ind}> {action} (+{int(reward*100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> " 

    input_str += "Current environment: " + look + " </s> " 
    input_str += "Current inventory: " + inventory + " </s> "
    
    if places:
        input_str +=  "Visited rooms: " + ", ".join(places)  + ' </s> '
        
    input_str += ' What action should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "") 
    label = sanitizeStr(label)
    return input_str, label


def compose_instance_v4_nohist(step_id, task_desc, returns_to_go, curr_action,
                        curr_obs, inventory, look, prev_action, prev_obs,
                        objects, places, recent_actions, recent_obs, recent_scores, recent_reward):

    ''' In this function the labels/predictions are the actions without storing subgoals and it does not keep the history of (action, observation).'''

    label = curr_action
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "
    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-1:], recent_actions[-1:], recent_reward[-1:]):
        input_str += f" <extra_id_{ind}> {action} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' What action should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")
    label = sanitizeStr(label)
    return input_str, label


def compose_instance_v4_nohist_2(mode, step_id, task_desc, returns_to_go, curr_action,
                                 curr_obs, inventory, look, prev_action, prev_obs,
                                 objects, places, recent_actions, recent_obs, recent_scores, recent_reward):
    assert mode == "fast_system"
    label = curr_action
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "
    input_str += "Action history: </s>"
    ind = 10
    for obs, action, reward in zip(recent_obs[-1:], recent_actions[-1:], recent_reward[-1:]):
        input_str += f" <extra_id_{ind}> {action} (+{int(reward * 100)}) --> {obs} | "
        ind -= 1
    input_str += " </s> "

    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "
    input_str += ' What action should you do next? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")
    label = sanitizeStr(label)
    return input_str, label


def compose_instance_v4_first_sgaslabel(mode, step_id, task_desc, inventory, look, 
                                   places,  recent_scores,
                                  curr_subgoal):
    ''' In this function the labels are the subgoals instead of the actions.'''

    assert mode == "fast_system"
    label = curr_subgoal
    input_str = task_desc + f" </s> Time: {step_id}; Score: {int(recent_scores[-1] * 100)}; </s> "
    input_str += "Current environment: " + look + " </s> "
    input_str += "Current inventory: " + inventory + " </s> "

    if places:
        input_str += "Visited rooms: " + ", ".join(places) + ' </s> '

    input_str += ' . You just have started this task. You did not do any actions yet. What is the first subtask you should do? </s> '
    input_str = sanitizeStr(input_str)
    input_str = input_str.replace("(that is open)", "")
    input_str = input_str.replace("(containing nothing)", "")

    return input_str, label


def recover_action(formalized_action):
    conversion_dict = [
        {"format_str": "0", "pattern": "CHOOSE(0)", "num_args": 0},
        {"format_str": "1", "pattern": "CHOOSE(1)", "num_args": 0},
        {"format_str": "look around", "pattern": "SEE()", "num_args": 0},
        {"format_str": "wait", "pattern": "WAIT()", "num_args": 0},
        # {"format_str": "wait1", "pattern": "WAIT()", "num_args": 0},
        {"format_str": "focus on {}", "pattern": r"^FOCUS\((.+)\)", "num_args": 1},
        {"format_str": "wait {}", "pattern": r"^WAIT\((.+)\)", "num_args": 1},
        {"format_str": "look at {}", "pattern": r"^LOOK\((.+)\)", "num_args": 1},
        {"format_str": "read {}", "pattern": r"^READ\((.+)\)", "num_args": 1},
        {"format_str": "pick up {}", "pattern": r"^PICK\((.+)\)", "num_args": 1},
        {"format_str": "pick up {}", "pattern": r"^PICKUP\((.+)\)", "num_args": 1},
        {"format_str": "pick up {}", "pattern": r"^PICK_UP\((.+)\)", "num_args": 1},
        {"format_str": "open door to {}", "pattern": r"^OPEN_DOOR\((.+)\)", "num_args": 1},
        {"format_str": "close door to {}", "pattern": r"^CLOSE_DOOR\((.+)\)", "num_args": 1},
        {"format_str": "open {}", "pattern": r"^OPEN\((.+)\)", "num_args": 1},
        {"format_str": "close {}", "pattern": r"^CLOSE\((.+)\)", "num_args": 1},
        {"format_str": "activate {}", "pattern": r"^ACTIVATE\((.+)\)", "num_args": 1},
        {"format_str": "deactivate {}", "pattern": r"^DEACTIVATE\((.+)\)", "num_args": 1},
        {"format_str": "activate {}", "pattern": r"^TURN_ON\((.+)\)", "num_args": 1},
        {"format_str": "deactivate {}", "pattern": r"^TURN_OFF\((.+)\)", "num_args": 1},
        {"format_str": "go to {}", "pattern": r"^GO\((.+)\)", "num_args": 1},
        {"format_str": "teleport to {}", "pattern": r"^TELEPORT\((.+)\)", "num_args": 1},
        {"format_str": "examine {}", "pattern": r"^EXAMINE\((.+)\)", "num_args": 1},
        {"format_str": "examine {}", "pattern": r"^OBSERVE\((.+)\)", "num_args": 1},
        {"format_str": "connect {} to {}", "pattern": r"^CONNECT\((.+), (.+)\)", "num_args": 2},
        {"format_str": "move {} to {}", "pattern": r"^MOVE\((.+), (.+)\)", "num_args": 2},
        {"format_str": "move {} to {}", "pattern": r"^PLACE\((.+), (.+)\)", "num_args": 2},
        {"format_str": "use {} on {}", "pattern": r"^USE\((.+), (.+)\)", "num_args": 2},
        {"format_str": "pour {} into {}", "pattern": r"^POUR\((.+), (.+)\)", "num_args": 2},
        {"format_str": "dunk {} into {}", "pattern": r"^DUNK\((.+), (.+)\)", "num_args": 2},
        {"format_str": "mix {}", "pattern": r"^MIX\((.+)\)", "num_args": 1},
        {"format_str": "drop {} in {}", "pattern": r"^DROP\((.+), (.+)\)", "num_args": 2},
        {"format_str": "drop {}", "pattern": r"^DROP\((.+)\)", "num_args": 1},
    ] 
     
    for item in conversion_dict:
        formal_action = action_conversion(formalized_action, **item)
        if formal_action:
            return formal_action
    print(f"{formalized_action} cannot be matched with any patterns.")
    return None  


def formalize_action(action):
    conversion_dict = [
        {"pattern": "0", "format_str": "CHOOSE(0)", "num_args": 0},
        {"pattern": "1", "format_str": "CHOOSE(1)", "num_args": 0},
        {"pattern": "look around", "format_str": "SEE()", "num_args": 0},
        {"pattern": "wait", "format_str": "WAIT()", "num_args": 0},
        {"pattern": "wait1", "format_str": "WAIT()", "num_args": 0},
        {"pattern": r"^focus on (.+)", "format_str": "FOCUS({})", "num_args": 1},
        {"pattern": r"^look at (.+)", "format_str": "LOOK({})", "num_args": 1},
        {"pattern": r"^read (.+)", "format_str": "READ({})", "num_args": 1},
        {"pattern": r"^pick up (.+)", "format_str": "PICK({})", "num_args": 1},
        {"pattern": r"^open door to (.+)", "format_str": "OPEN_DOOR({})", "num_args": 1},
        {"pattern": r"^close door to (.+)", "format_str": "CLOSE_DOOR({})", "num_args": 1},
        {"pattern": r"^open (.+)", "format_str": "OPEN({})", "num_args": 1},
        {"pattern": r"^close (.+)", "format_str": "CLOSE({})", "num_args": 1},
        {"pattern": r"^activate (.+)", "format_str": "ACTIVATE({})", "num_args": 1},
        {"pattern": r"^deactivate (.+)", "format_str": "DEACTIVATE({})", "num_args": 1},
        {"pattern": r"^go to (.+)", "format_str": "GO({})", "num_args": 1},
        {"pattern": r"^teleport to (.+)", "format_str": "TELEPORT({})", "num_args": 1},
        {"pattern": r"^examine (.+)", "format_str": "EXAMINE({})", "num_args": 1},
        {"pattern": r"^connect (.+) to (.+)", "format_str": "CONNECT({}, {})", "num_args": 2},
        {"pattern": r"^move (.+) to (.+)", "format_str": "MOVE({}, {})", "num_args": 2},
        {"pattern": r"^use (.+) on (.+)", "format_str": "USE({}, {})", "num_args": 2},
        {"pattern": r"^pour (.+) into (.+)", "format_str": "POUR({}, {})", "num_args": 2},
        {"pattern": r"^pour (.+) in (.+)", "format_str": "POUR({}, {})", "num_args": 2},
        {"pattern": r"^dunk (.+) into (.+)", "format_str": "DUNK({}, {})", "num_args": 2},
        {"pattern": r"^mix (.+)", "format_str": "MIX({})", "num_args": 1},
        {"pattern": r"^drop (.+) in (.+)", "format_str": "DROP({}, {})", "num_args": 2},
        {"pattern": r"^drop (.+)", "format_str": "DROP({})", "num_args": 1},
        {"pattern": "end subgoal", "format_str": "END_SG()", "num_args": 0},
    ]     
    for item in conversion_dict:
        formal_action = action_conversion(action, **item)
        if formal_action:
            return formal_action
    
    return None  


if __name__ == "__main__":
    actions = [
        "focus on metal pot", 
        "look around", 
        "open door to art studio",
        "move metal pot to stove",
        "pour cup1 in inventory in art studio in cup2",
        "mix obj1 and obj2 and obj3",
        "drop light",
        "deactivate sink",
        "wait1",
        "end subgoal"
    ]
    
    for action in actions:
        f = formalize_action(action)
        r = recover_action(f)
        print(f"{action} --> {f}  ---> {r}")
            