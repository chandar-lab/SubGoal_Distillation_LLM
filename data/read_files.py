import json
import os

# Opening JSON file
""" In this file, the train, test, and val data generated already for each task 
are combined in one train, test, and val file for training the model. """

dir = "../data_v4_sg"
modes = ["train", "test"]

train_data = []
val_data = []
test_data = []

for mode in modes:
    print("*************** " + mode + " *********************")
    data = []
    i = 0
    for file_name in [
        file for file in os.listdir(f"{dir}/data_dir/{mode}/") if file.endswith(".json")
    ]:
        i += 1
        print(file_name, "----", i)
        with open(f"{dir}/data_dir/{mode}/" + file_name) as json_file:
            for line in json_file:
                data.append(json.loads(line))
            print(data[0])

    if mode == "train":
        train_data = data
    elif mode == "test":
        test_data = data

mode = "val"
print("*************** " + mode + " *********************")
data = []
i = 0
for file_name in [
    file for file in os.listdir(f"{dir}/data_dir/{mode}/") if file.endswith(".val.json")
]:
    i += 1
    print(file_name, "----", i)
    with open(f"{dir}/data_dir/{mode}/" + file_name) as json_file:
        for line in json_file:
            data.append(json.loads(line))
        print(data[0])

if mode == "val":
    val_data = data


with open(f"{dir}/fast_system.train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(f"{dir}/fast_system.val.jsonl", "w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{dir}/fast_system.val.mini.jsonl", "w") as f:
    import random

    random.seed(1)
    random.shuffle(val_data)
    val_data = val_data[:10000]
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"{dir}/fast_system.test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
