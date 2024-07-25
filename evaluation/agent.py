import argparse
import os
import re
import time
import torch
import random
import copy
from scienceworld import ScienceWorldEnv
import json
from tqdm import trange
import sys

from data.data_utils import (
    clean,
    add_current_place,
    add_current_objects,
    sanitizeStr,
    formalize_action,
    recover_action,
)
from data.data_utils import (
    compose_instance_v4,
    compose_instance_v4_sg,
    compose_instance_v4_nohist,
    compose_instance_v4_sgaslabel,
    compose_instance_v4_sg_nohist,
    compose_instance_v4_nohist_2,
    get_real_task_id,
    compose_instance_v4_first_sgaslabel,
)
from subgoals.sg_generating import sg_generator
from eval_utils import (
    load_model,
    findValidActionNew,
    load_variation,
    get_model_output,
    load_model_sg,
    load_model_Firstsg,
)
from generate_sg_eval import sg_generator_eval

import logging
from logging import INFO, WARN
import numpy as np
from datasets import load_dataset, load_metric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_file_name(args, task_num):
    if len(args["output_path"]) > 0:
        args["output_path"] += "/"

        # Make path if it doesn't exist
        if not os.path.exists(args["output_path"]):
            os.makedirs(args["output_path"])
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed


def creat_semi_random_sg(first_subgoal, PossibleObjects, locations):
    """this function change the objects or locations in the subgoals to generates semi-random subgoals."""
    params_list = []
    st = first_subgoal.find("(")
    end = first_subgoal.find(")")
    params = first_subgoal[st + 1 : end]
    if "," in params:
        camma = first_subgoal.find(",")
        param1 = first_subgoal[st:camma]
        param2 = first_subgoal[camma + 1 : end]
        params_list.append(param1)
        params_list.append(param2)
    else:
        params_list.append(params)

    for i in range(len(params_list)):
        if params_list[i].strip().strip(",") in locations:
            params_list[i] = random.choice(locations)
        else:
            params_list[i] = random.choice(PossibleObjects)

    replace_part = first_subgoal[:st] + "(" + params_list[0]
    if len(params_list) == 2:
        replace_part += ", " + params_list[1]

    replace_part += first_subgoal[end:]

    return replace_part


def eval(args, task_num, logger):
    """this function is for interactive evaluation"""
    ## print the parameters
    print(args.keys())
    print(
        "LM parameters : mode_sg (lm or rand), sg_(using sg or not), hist_ (using hist or not): "
    )
    print(f"mode_sg is {args['mode_sg']} and this agent uses sg is {args['sg_']}")

    if args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4  # compose_instance_v4
    elif args["compose_mode"] == "v4_nohist":
        compose_instance = compose_instance_v4_nohist
    elif args["compose_mode"] == "v4_nohist_v2":
        compose_instance = compose_instance_v4_nohist_2
    elif args["compose_mode"] == "v4_with_sg":
        compose_instance = compose_instance_v4_sg
    elif args["compose_mode"] == "v4_with_sg_nohist":
        compose_instance = compose_instance_v4_sg_nohist
    if (
        args["compose_mode_sg"] == "v4_for_sg"
    ):  ## we use this composition to create input instruction for the sg moel
        compose_instance_sg = compose_instance_v4_sgaslabel

    # Initialize environment
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit=args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args["simplification_str"])

    ## load all of the subgoals
    all_sg = np.load("all_sg_array.npy")
    locations = [
        "kitchen",
        "bathroom",
        "workshop",
        "art studio",
        "greenhouse",
        "outside",
    ]

    ## load models: executer, controller and first subgoal generator
    lm_model, tokenizer, sbert_model = load_model(args, device)
    lm_model_sg, tokenizer_sg, sbert_model_sg = load_model_sg(args, device)
    lm_model_Fsg, tokenizer_Fsg, sbert_model_Fsg = load_model_Firstsg(args, device)
    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)

    print(" task id and task name, variations: ", task_num, taskName, variations)
    scores = []
    all_scores = []

    task_idx_real = get_real_task_id(taskName)
    file_name = f"/test/fast_system.{task_idx_real}.test.json"
    raw_datasets = load_dataset("json", data_files=file_name)
    variation_ids = np.array(raw_datasets["train"]["variation_id"])
    uni_var = set(variation_ids)
    variations_intest = list(map(lambda x: x, uni_var))

    for variation in variations_intest[:10]:
        print("---- Var is:", variation)
        env.load(
            taskName, int(variation), args["simplification_str"], generateGoldPath=True
        )
        task_description = env.taskdescription()[18:]
        recent_actions = ["look around"]
        recent_obs = ["N/A"]
        recent_scores = [
            0.0,
        ]
        recent_reward = [0.0]
        places = []
        objects = []
        obs, info = env.reset()
        prev_obs = "N/A"
        prev_action = "look around"

        done = False
        score = 0.0
        last_score = 0.0
        step = 0
        mode = args["mode"]

        ## to avoid loop trap
        max_steps = args["env_step_limit"] * 2

        ### you can add first sg generator here as well

        while not done:

            if step == 0:
                if args["sg_"]:
                    print("---- First sg")
                    task_name_id = taskName + "_" + str(task_num)
                    steps, subgoals = sg_generator(
                        env.getGoldActionSequence(),
                        env.taskdescription(),
                        variation,
                        env.getGoldActionSequence(),
                        None,
                        task_name_id,
                    )
                    modified_sg = []

                    ## first sg generator
                    if args["mode_sg"] == "rand" and step % 10 == 0:
                        first_subgoal = random.choice(all_sg)
                    elif args["mode_sg"] == "semi-rand" and step % 10 == 0:
                        input_first_sg, _ = compose_instance_v4_first_sgaslabel(
                            mode=mode,
                            step_id=step + 1,
                            task_desc=task_description,
                            inventory=info["inv"],
                            look=info["look"],
                            places=places,
                            recent_scores=recent_scores,
                            curr_subgoal=None,
                        )
                        first_subgoal = (
                            get_model_output(
                                args,
                                input_first_sg,
                                tokenizer_Fsg,
                                lm_model_Fsg,
                                device,
                                logger,
                            )[0]
                            .strip()
                            .strip(".")
                        )
                        first_subgoal = creat_semi_random_sg(
                            first_subgoal, env.getPossibleObjects(), locations
                        )
                    else:
                        input_first_sg, _ = compose_instance_v4_first_sgaslabel(
                            mode=mode,
                            step_id=step + 1,
                            task_desc=task_description,
                            inventory=info["inv"],
                            look=info["look"],
                            places=places,
                            recent_scores=recent_scores,
                            curr_subgoal=None,
                        )
                        first_subgoal = (
                            get_model_output(
                                args,
                                input_first_sg,
                                tokenizer_Fsg,
                                lm_model_Fsg,
                                device,
                                logger,
                            )[0]
                            .strip()
                            .strip(".")
                        )

                    if args["first_sg_rand"]:
                        first_subgoal = random.choice(all_sg)

                    subgoal_index = 0
                    curr_subgoal = first_subgoal
                    prev_subgoal = first_subgoal
                    curr_subgoal = findValidActionNew(
                        " ",
                        env,
                        info["look"],
                        recent_actions,
                        sbert_model,
                        logger,
                        5,
                        True,
                        taskName,
                        variation,
                        curr_subgoal,
                        task_description,
                    )

            input_str = ""

            add_current_place(obs, info["look"], places)
            add_current_objects(task_num, info["look"], objects, limit=20)

            # Note that the agent is allowed to know the score changes.
            returns_to_go = 1.0 - float(info["score"]) * 0.01
            returns_to_go = round(returns_to_go, 2)

            mode = args["mode"]
            logger.info("Mode: " + mode)

            if not args["sg_"]:
                input_str, _ = compose_instance(
                    mode=mode,
                    step_id=step + 1,
                    task_desc=task_description,
                    returns_to_go=returns_to_go,
                    curr_action=None,
                    curr_obs=obs,
                    inventory=info["inv"],
                    look=info["look"],
                    prev_action=prev_action,
                    prev_obs=prev_obs,
                    objects=objects,
                    places=places,
                    recent_actions=recent_actions,
                    recent_obs=recent_obs,
                    recent_scores=recent_scores,
                    recent_reward=recent_reward,
                )

            else:

                ### add subgoals
                if step == 0:
                    modified_sg.append(curr_subgoal)
                else:
                    prev_subgoal = curr_subgoal

                    if args["mode_sg"] == "rand" and step % 10 == 0:
                        curr_subgoal = random.choice(all_sg)
                        print(
                            "curr_subgoal is",
                            curr_subgoal,
                            "last action: ",
                            recent_actions[-1],
                            "last obs: ",
                            recent_obs[-1],
                            "last reward: ",
                            recent_reward[-1],
                        )

                    elif args["mode_sg"] == "semi-rand" and step % 10 == 0:
                        input_str_sg, _ = compose_instance_sg(
                            mode=mode,
                            step_id=step + 1,
                            task_desc=task_description,
                            returns_to_go=returns_to_go,
                            curr_action=None,
                            curr_obs=obs,
                            inventory=info["inv"],
                            look=info["look"],
                            prev_action=prev_action,
                            prev_obs=prev_obs,
                            objects=objects,
                            places=places,
                            recent_actions=recent_actions,
                            recent_obs=recent_obs,
                            recent_scores=recent_scores,
                            recent_reward=recent_reward,
                            subgoals=modified_sg,
                            curr_subgoal=None,
                            last_sg=prev_subgoal,
                        )

                        curr_subgoal = (
                            get_model_output(
                                args,
                                input_str_sg,
                                tokenizer_sg,
                                lm_model_sg,
                                device,
                                logger,
                            )[0]
                            .strip()
                            .strip(".")
                        )
                        curr_subgoal = creat_semi_random_sg(
                            curr_subgoal, env.getPossibleObjects(), locations
                        )
                        print(
                            "curr_subgoal is",
                            curr_subgoal,
                            "last action: ",
                            recent_actions[-1],
                            "last obs: ",
                            recent_obs[-1],
                            "last reward: ",
                            recent_reward[-1],
                        )

                    else:
                        input_str_sg, _ = compose_instance_sg(
                            mode=mode,
                            step_id=step + 1,
                            task_desc=task_description,
                            returns_to_go=returns_to_go,
                            curr_action=None,
                            curr_obs=obs,
                            inventory=info["inv"],
                            look=info["look"],
                            prev_action=prev_action,
                            prev_obs=prev_obs,
                            objects=objects,
                            places=places,
                            recent_actions=recent_actions,
                            recent_obs=recent_obs,
                            recent_scores=recent_scores,
                            recent_reward=recent_reward,
                            subgoals=modified_sg,
                            curr_subgoal=None,
                            last_sg=prev_subgoal,
                        )
                        curr_subgoal = (
                            get_model_output(
                                args,
                                input_str_sg,
                                tokenizer_sg,
                                lm_model_sg,
                                device,
                                logger,
                            )[0]
                            .strip()
                            .strip(".")
                        )

                        ### validity of sg:
                        print(
                            "curr_subgoal is",
                            curr_subgoal,
                            "last action: ",
                            recent_actions[-1],
                            "last obs: ",
                            recent_obs[-1],
                            "last reward: ",
                            recent_reward[-1],
                        )

                if clean(curr_subgoal).strip().strip(".") != clean(
                    modified_sg[subgoal_index]
                ).strip().strip("."):
                    subgoal_index += 1
                    modified_sg.append(curr_subgoal)

                input_str, _ = compose_instance(
                    mode=mode,
                    step_id=step + 1,
                    task_desc=task_description,
                    returns_to_go=returns_to_go,
                    curr_action=None,
                    curr_obs=obs,
                    inventory=info["inv"],
                    look=info["look"],
                    prev_action=prev_action,
                    prev_obs=prev_obs,
                    objects=objects,
                    places=places,
                    recent_actions=recent_actions,
                    recent_obs=recent_obs,
                    recent_scores=recent_scores,
                    recent_reward=recent_reward,
                    subgoals=modified_sg,
                    curr_subgoal=curr_subgoal,
                )

            input_str = sanitizeStr(input_str)
            logger.info("InputStr: " + input_str)
            predStrs = get_model_output(
                args, input_str, tokenizer, lm_model, device, logger
            )
            predStrs_org = predStrs.copy()

            prev_obs = obs
            if len(predStrs) == 0:
                predStrs = [predStrs_org[0]]

            # Get valid actions at this point
            if "end subgoal" in predStrs or "END_SG()" in predStrs:
                action = predStrs
                print(" ---- action is: --- ", action)
                obs, reward, done, info = prev_obs, 0, False, info
            else:

                action = clean(sanitizeStr(predStrs[0])).strip()
                obs, reward, done, info = env.step(action)
                if (
                    "No known action matches that input" in obs
                    or "Unknown action" in obs
                ):
                    print("BBBBBBAAAAAADDDDD format")
                    action = findValidActionNew(
                        predStrs,
                        env,
                        info["look"],
                        recent_actions,
                        sbert_model,
                        logger,
                        5,
                        False,
                        taskName,
                    )
                    obs, reward, done, info = env.step(action)

            score = info["score"]
            prev_action = action
            reward = score - last_score
            recent_reward.append(reward / 100)
            recent_scores.append(score / 100)
            recent_actions.append(action)
            recent_obs.append(obs)
            print(step, done, score, "action is:", action)

            if score < 0:
                print("score is negative")
                if args["no_stop"]:
                    done = True
                    score = last_score
                else:
                    done = True
                    score = 0
            last_score = score

            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + sanitizeStr(obs))
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            print(step, done)
            if (step >= max_steps) or done:

                break

            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            if not args["sg_"]:
                if len(recent_actions) >= 10 and len(set(recent_actions[-10:])) == 2:
                    logger.info(
                        "Many recent actions in history are the same -- model is likely in a loop, stopping early."
                    )
                    break
            else:
                if len(set(recent_scores[-50:])) == 1 and len(recent_scores) > 50:
                    logger.info(
                        "Many recent actions do not change the score -- model is likely in a loop, stopping early."
                    )
                    break

        # Store results
        env.storeRunHistory(
            variation, notes={"mode": args["mode"], "lm": str(args["lm_path"])}
        )
        env.saveRunHistoriesBufferIfFull(
            filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"]
        )

        scores.append(score)
        print("scores are:", scores)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))

        time.sleep(2)

        avg = sum(scores) / len(scores)
        logger.info("Average score: " + str(avg))
        print("Average score: ", avg)

        f = open(filenameOutPrefixSeed + "-score.txt", "a")
        f.write(
            "\n"
            + "Task name:"
            + taskName
            + "Scores: "
            + str(scores)
            + " Average score: "
            + str(avg)
            + " Args: "
            + str(args)
            + "\n"
        )
        f.close()

    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")

    return avg


def eval_test(args, task_num, logger):
    """this function is for exact matching for action given subgoals"""
    print(args.keys())
    print(
        "LM parameters : mode_sg (lm or rand), sg_(using sg or not), hist_ (using hist or not): "
    )
    print(args["mode_sg"], args["sg_"])

    if args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4  # compose_instance_v4
    elif args["compose_mode"] == "v4_nohist":
        compose_instance = compose_instance_v4_nohist
    elif args["compose_mode"] == "v4_nohist_v2":
        compose_instance = compose_instance_v4_nohist_2
    elif args["compose_mode"] == "v4_with_sg":
        compose_instance = compose_instance_v4_sg
    elif args["compose_mode"] == "v4_with_sg_nohist":
        compose_instance = compose_instance_v4_sg_nohist
    if (
        args["compose_mode_sg"] == "v4_for_sg"
    ):  ## we use this composition to create input instruction for the sg moel
        compose_instance_sg = compose_instance_v4_sgaslabel

    ## load all subgoals
    all_sg = np.load("all_sg_array.npy")
    all_percentage = []
    all_scores = []
    dir = args["dir"]
    print([file for file in os.listdir(f"{dir}") if file.endswith(".json")])
    for files in [file for file in os.listdir(f"{dir}") if file.endswith(".json")]:

        data_files = f"{dir}" + files
        print(data_files)
        raw_datasets = load_dataset("json", data_files=data_files)
        inputs = raw_datasets["train"]["input"]
        targets = raw_datasets["train"]["target"]
        task_ids = raw_datasets["train"]["task_id"]
        variation_ids = np.array(raw_datasets["train"]["variation_id"])
        task_real_ids = raw_datasets["train"]["task_real_id"]
        uni_var = set(variation_ids)
        variations = list(map(lambda x: x, uni_var))

        # Initialize environment
        env = ScienceWorldEnv("", args["jar_path"], envStepLimit=args["env_step_limit"])
        taskNames = env.getTaskNames()
        task_num = task_ids[0]
        taskName = taskNames[task_num]
        env.load(taskName, 0, args["simplification_str"])
        lm_model, tokenizer, sbert_model = load_model(args, device)
        lm_model_sg, tokenizer_sg, sbert_model_sg = load_model_sg(args, device)
        filenameOutPrefixSeed = get_file_name(args, task_num)
        print(" task id and task name, variations: ", task_num, taskName, variations)

        scores = []
        all_actions = []
        real_actions = []
        for variation in variations:

            indexes = np.where(variation_ids == variation)[0]

            print("---- Var is:", variation)
            env.load(
                taskName,
                int(variation),
                args["simplification_str"],
                generateGoldPath=True,
            )
            task_description = env.taskdescription()[18:]
            recent_actions = ["look around"]
            recent_obs = ["N/A"]
            recent_scores = [
                0.0,
            ]
            recent_reward = [0.0]
            places = []
            objects = []
            obs, info = env.reset()

            prev_obs = "N/A"
            prev_action = "look around"
            done = False
            score = 0.0
            last_score = 0.0
            step = 0

            max_steps = args["env_step_limit"] * 2

            if args["sg_"]:
                modified_sg = []

            for i in range(len(indexes)):

                input_str = ""
                add_current_place(obs, info["look"], places)
                add_current_objects(task_num, info["look"], objects, limit=20)
                returns_to_go = 1.0 - float(info["score"]) * 0.01
                returns_to_go = round(returns_to_go, 2)

                mode = args["mode"]
                logger.info("Mode: " + mode)

                if not args["sg_"]:
                    input_str = inputs[indexes[i]]
                else:
                    if args["mode_sg"] == "lm":
                        input_str = inputs[indexes[i]]
                    elif args["mode_sg"] == "rand":
                        curr_subgoal = random.choice(all_sg)
                        prev_subgoal = curr_subgoal
                        if len(modified_sg) == 0:
                            modified_sg.append(curr_subgoal)
                        elif modified_sg[-1] != curr_subgoal:
                            modified_sg.append(curr_subgoal)

                        input_str, _ = compose_instance(
                            mode=mode,
                            step_id=step + 1,
                            task_desc=task_description,
                            returns_to_go=returns_to_go,
                            curr_action=None,
                            curr_obs=obs,
                            inventory=info["inv"],
                            look=info["look"],
                            prev_action=prev_action,
                            prev_obs=prev_obs,
                            objects=objects,
                            places=places,
                            recent_actions=recent_actions,
                            recent_obs=recent_obs,
                            recent_scores=recent_scores,
                            recent_reward=recent_reward,
                            subgoals=modified_sg,
                        )

                input_str = sanitizeStr(input_str)
                logger.info("InputStr: " + input_str)
                predStrs = get_model_output(
                    args, input_str, tokenizer, lm_model, device, logger
                )
                prev_obs = obs
                action = findValidActionNew(
                    predStrs, env, info["look"], recent_actions, sbert_model, logger
                )
                obs, reward, done, info = env.step(action)
                score = info["score"]
                prev_action = action
                reward = score - last_score
                recent_reward.append(reward / 100)
                recent_scores.append(score / 100)
                recent_actions.append(action)
                recent_obs.append(obs)
                real_actions.append(targets[indexes[i]])
                all_actions.append(action)

                if score < 0:
                    if args["no_stop"]:
                        done = True
                        score = last_score
                    else:
                        done = True
                        score = 0
                last_score = score

                logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
                logger.info("Obs: " + sanitizeStr(obs))
                logger.info(f"Score: {score}")
                logger.info("")

                step += 1
                if (step >= max_steps) or done:
                    break

                logger.info("Recent Actions: " + str(recent_actions))

                # Early stopping if we're in a loop
                if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                    logger.info(
                        "Many recent actions in history are the same -- model is likely in a loop, stopping early."
                    )
                    break

            # Store results
            env.storeRunHistory(
                variation, notes={"mode": args["mode"], "lm": str(args["lm_path"])}
            )
            env.saveRunHistoriesBufferIfFull(
                filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"]
            )

            scores.append(score)

            logger.info("Run completed...")
            logger.info("Scores: " + str(scores))

            time.sleep(2)

        avg = sum(scores) / len(scores)
        all_scores.append(avg)
        logger.info("Average score: " + str(avg))
        print("Average score is:", avg)

        f = open(filenameOutPrefixSeed + "-score.txt", "a")
        f.write(
            "\n"
            + "Task name:"
            + taskName
            + "Scores: "
            + str(scores)
            + " Average score: "
            + str(avg)
            + " Args: "
            + str(args)
            + "\n"
        )
        f.close()

        logger.info("Shutting down server...")

        matching_count = sum(1 for a, b in zip(all_actions, real_actions) if a == b)
        percentage = (matching_count / len(all_actions)) * 100
        logger.info("percentage of exact matching is:" + str(percentage))
        print("percentage is:", percentage)
        logger.info("Completed.")
        all_percentage.append(percentage)
        print("all percentages are: ", all_percentage)
        print("all scores are: ", all_scores)
        np.save(
            f'{args["output_path"]}/all_percentage_{args["seed"]}',
            np.array(all_percentage),
        )
        np.save(
            f'{args["output_path"]}/all_scores_{args["seed"]}', np.array(all_scores)
        )


def eval_test_sg(args, task_num, logger):
    """this function is for exact matching for sg"""
    print(args.keys())
    print(
        "LM parameters : mode_sg (lm or rand), sg_(using sg or not), hist_ (using hist or not): "
    )
    print(args["mode_sg"], args["sg_"])

    if args["compose_mode"] == "v4":
        compose_instance = compose_instance_v4  # compose_instance_v4
    elif args["compose_mode"] == "v4_nohist":
        compose_instance = compose_instance_v4_nohist
    elif args["compose_mode"] == "v4_nohist_v2":
        compose_instance = compose_instance_v4_nohist_2
    elif args["compose_mode"] == "v4_with_sg":
        compose_instance = compose_instance_v4_sg
    elif args["compose_mode"] == "v4_with_sg_nohist":
        compose_instance = compose_instance_v4_sg_nohist
    if (
        args["compose_mode_sg"] == "v4_for_sg"
    ):  ## we use this composition to create input instruction for the sg moel
        compose_instance_sg = compose_instance_v4_sgaslabel

    ## load all subgoals
    all_sg = np.load("all_sg_array.npy")
    all_percentage = []
    dir = args["dir"]
    for files in [file for file in os.listdir(f"{dir}") if file.endswith(".json")]:

        logger.info(f"task file: {files}")
        data_files = f"{dir}" + files
        print(data_files)
        raw_datasets = load_dataset("json", data_files=data_files)
        inputs = raw_datasets["train"]["input"]
        targets = raw_datasets["train"]["target"]
        lm_model_sg, tokenizer_sg, sbert_model_sg = load_model_sg(args, device)
        task_ids = raw_datasets["train"]["task_id"]
        variation_ids = np.array(raw_datasets["train"]["variation_id"])
        task_real_ids = raw_datasets["train"]["task_real_id"]
        uni_var = set(variation_ids)
        variations = list(map(lambda x: x, uni_var))
        env = ScienceWorldEnv("", args["jar_path"], envStepLimit=args["env_step_limit"])
        taskNames = env.getTaskNames()
        task_num = task_ids[0]
        taskName = taskNames[task_num]
        env.load(taskName, 0, args["simplification_str"])
        lm_model_sg, tokenizer_sg, sbert_model_sg = load_model_sg(args, device)
        print(" task id and task name, variations: ", task_num, taskName, variations)
        logger.info(f"task NAME : {taskName}")
        all_sg = []
        real_sg = []

        for variation in variations:

            print("---- Var is:", variation)
            indexes = np.where(variation_ids == variation)[0]

            for i in range(len(indexes)):
                input_str_sg = inputs[indexes[i]]
                curr_subgoal = get_model_output(
                    args, input_str_sg, tokenizer_sg, lm_model_sg, device, logger
                )[0]
                print(
                    "curr_subgoal is", curr_subgoal, "curr target", targets[indexes[i]]
                )
                all_sg.append(clean(curr_subgoal).strip().strip("."))
                real_sg.append(clean(targets[indexes[i]]).strip().strip("."))
                matching_count = sum(1 for a, b in zip(all_sg, real_sg) if a == b)
                percentage = (matching_count / len(real_sg)) * 100
                print(f"percentage for var= {variation} is: ", percentage)

        matching_count = sum(1 for a, b in zip(all_sg, real_sg) if a == b)
        percentage = (matching_count / len(real_sg)) * 100
        print("percentage is:", percentage)
        all_percentage.append(percentage)

        logger.info(f"Variation: {variation}")
        logger.info("")

        logger.info("Recent sgs: " + str(all_sg))
        logger.info("percentage of exact matching is:" + str(percentage))
        logger.info("Completed.")

        print("all percentages are: ", all_percentage)
        np.save(
            f'{args["output_path"]}/all_percentage_{args["seed"]}',
            np.array(all_percentage),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    debug = False
    if not debug:
        parser.add_argument("--jar_path", type=str)
        parser.add_argument("--task_nums", default="17")  # use comma to split
        parser.add_argument("--env_step_limit", type=int, default=100)
        parser.add_argument(
            "--lm_path", default="path to action generator model (executor)"
        )
        parser.add_argument(
            "--sg_lm_path", help="path to sg generator checkpoint model (controller)"
        )
        parser.add_argument(
            "--Fsg_lm_path", default="path to first sg checkpoint model"
        )
        parser.add_argument("--simplification_str", default="easy")
        parser.add_argument("--beams", type=int, default=5)
        parser.add_argument("--max_episode_per_file", type=int, default=9999)
        parser.add_argument("--mode", default="fast_system")
        parser.add_argument(
            "--set",
            default="test",
            help="which dataset is going to be used, train or test",
        )
        parser.add_argument("--output_path", default="/test/test_fsg_int_flanl_3/")
        parser.add_argument(
            "--dir", default="/data/data_v4_all_first_sg/data_dir/test/"
        )
        parser.add_argument(
            "--compose_mode", default="v4_with_sg", help="prompt type for executer"
        )
        parser.add_argument(
            "--compose_mode_sg",
            type=str,
            default="v4_for_sg",
            help="prompt type for controller",
        )
        parser.add_argument("--model_parallelism_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_input_len", type=int, default=2024)
        parser.add_argument("--cut_off", action="store_true", default=True)
        parser.add_argument("--no_stop", action="store_true", default=False)
        parser.add_argument("--sbert", action="store_true", default=False)
        parser.add_argument(
            "--mode_sg",
            default="rand",
            help="the mode of generating sg; it can be either 'lm', 'rand', 'semi-rand'",
        )
        parser.add_argument(
            "--rand_percentage",
            type=int,
            default=1,
            help="it is for semi-random, or random, the persentage of chossing a wrong subgoal",
        )
        parser.add_argument(
            "--sg_",
            action="store_true",
            help="it can be either false or true. False means no sg in the the prompt",
        )
        parser.add_argument(
            "--first_sg_rand",
            action="store_true",
            help="it can be either false or true. False means no random selection for first sg",
        )

        parser.add_argument(
            "--test_mode",
            type=str,
            default="interactive",
            help=" 'sg' means using sg but not intractive, 'interactive', 'em' --> exact matching",
        )
    else:
        parser.add_argument("--jar_path", type=str)
        parser.add_argument("--task_nums", default="28")  # use comma to split
        parser.add_argument("--env_step_limit", type=int, default=100)
        parser.add_argument(
            "--lm_path", default="fast_agent/model_ckpts/flan_large_0402/checkpoint-300"
        )
        parser.add_argument("--simplification_str", default="easy")
        parser.add_argument("--beams", type=int, default=5)
        parser.add_argument("--max_episode_per_file", type=int, default=9999)
        parser.add_argument("--mode", default="fast_system")
        parser.add_argument("--set", default="test")
        parser.add_argument("--output_path", default="logs/test_fl-v1-300-bm=5")
        parser.add_argument("--compose_mode", default="v1_1")
        parser.add_argument("--model_parallelism_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max_input_len", type=int, default=1024)
        parser.add_argument("--cut_off", action="store_true", default=True)
        parser.add_argument("--sbert", action="store_true", default=False)
        parser.add_argument("--no_stop", action="store_true", default=True)
        parser.add_argument("--sg_lm_path", default="sg_lm_model")

    args = parser.parse_args()
    params = vars(args)
    return params


def init_logger(args, task_num, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num)
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s\t] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime("%Y-%m-%d_%H-%M", time.localtime(now / 1000))
        filename = f"{filenameOutPrefixSeed}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    print(args)

    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args["test_mode"] == "interactive":
        task_nums = args["task_nums"].split(",")
        all_scores = []
        for task_num in task_nums:
            print("task num is:", task_num)
            logger = init_logger(args, task_num)
            logger.info(args)
            score = eval(args, int(task_num), logger)
            all_scores.append(score)
            print("all scores are:", all_scores)
            np.save(
                f'{args["output_path"]}/all_scores_{args["seed"]}', np.array(all_scores)
            )

    if args["test_mode"] == "sg":
        task_nums = args["task_nums"].split(",")
        all_scores = []
        for task_num in task_nums:
            logger = init_logger(args, task_num)
            logger.info(args)
            eval_test_sg(args, int(task_num), logger)

    if args["test_mode"] == "em":
        task_nums = args["task_nums"].split(",")
        all_scores = []
        for task_num in task_nums:
            logger = init_logger(args, task_num)
            logger.info(args)
            eval_test(args, int(task_num), logger)


if __name__ == "__main__":
    main()
