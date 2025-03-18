import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    
    prompt_file_path = f"generated_data/{dataset_name}/prompt4generation"

    raw_prompt_files = os.listdir(prompt_file_path)
    raw_prompt_files = [
        i for i in raw_prompt_files if ".jsonl" in i and "_combination" not in i and "test" not in i and "valid" not in i]

    unique_id = set()
    output_file = f"generated_data/{dataset_name}/prompt4generation/{dataset_name}_combination.jsonl"
    output_file_writter = open(output_file, 'w', encoding='utf8')

    for input_file in raw_prompt_files:
        with open(os.path.join(prompt_file_path, input_file), 'r', encoding='utf8') as f:
            for line in f:
                if not line:
                    print(line)
                    continue
                raw_line_data = json.loads(line)
                if raw_line_data["customer_id"] in unique_id:
                    raise KeyError("key conflict")
                unique_id.add(raw_line_data["customer_id"])
                output_file_writter.write(json.dumps(raw_line_data)+"\n")