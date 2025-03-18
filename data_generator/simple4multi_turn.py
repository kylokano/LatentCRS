import json
import random

if __name__ == "__main__":
    random.seed(1024)

    user_sample_number = 500
    dataset_name_list = ["ml-1m", "cds", "VedioGames"]

    all_data = {}
    all_items = []
    for dataset_name in dataset_name_list:
        all_test_data = []
        with open(f"generated_data/{dataset_name}/processed_data/{dataset_name}_raw_seedtest.jsonl", "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                all_test_data.append(json.loads(line))
                all_items.extend(json.loads(line)["history"])
                all_items.extend([json.loads(line)["target"]])
            sampled_data = random.sample(all_test_data, user_sample_number)
        all_data[dataset_name] = sampled_data
        all_items = list(set(all_items))
        
    for dataset_name, sampled_data in all_data.items():
        with open(f"generated_data/{dataset_name}/processed_data/{dataset_name}_raw_seedtest_sample_{user_sample_number}.jsonl", "w") as f:
            for data in sampled_data:
                f.write(json.dumps(data) + "\n")