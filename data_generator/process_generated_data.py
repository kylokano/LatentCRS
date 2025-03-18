import json
import os
import random
import argparse
import logging
from typing import List, Dict, Tuple

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Process generated data from LLM')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, default="gemini", help='Model name')
    parser.add_argument('--data_split', type=str, required=True, 
                       help='data split (combination/valid/test)')
    return parser.parse_args()

def setup_paths(args) -> Tuple[str, str, str]:
    raw_data_path = f"generated_data/{args.dataset}/prompt4generation"
    generated_data_path = f"generated_data/{args.dataset}/{args.model}_generated"
    output_path = f"generated_data/{args.dataset}/processed_data"
    
    os.makedirs(generated_data_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    return raw_data_path, generated_data_path, output_path

def get_datanames(dataset_name: str, data_split: str, generated_files: List[str]) -> Tuple[str, str]:
    if data_split == "combination":
        raw_data_name = f"{dataset_name}_combination.jsonl"
    else:
        raw_data_name = f"{dataset_name}_raw_seed{data_split}.jsonl"
    
    startstr = raw_data_name.replace(".jsonl", "") + "_output"
    generated_data_name = next((f for f in generated_files if f.startswith(startstr)), None)
    
    if not generated_data_name:
        raise KeyError(f"No matching generated file found for {raw_data_name}")
    
    return raw_data_name, generated_data_name

def load_responses(generated_data_path: str, generated_data_name: str) -> Dict[str, str]:
    responses_dict = {}
    error_count = 0
    
    with open(os.path.join(generated_data_path, generated_data_name), 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            try:
                response = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
                responses_dict[data["customer_id"]] = response
            except Exception as e:
                logging.error(f"Error processing response: {e}")
                logging.debug(f"Problematic data: {data}")
                error_count += 1
    
    logging.info(f"Total response processing errors: {error_count}")
    return responses_dict

def process_raw_data(raw_data_path: str, raw_data_name: str, 
                    responses_dict: Dict[str, str], random_seed: str) -> List[Dict]:
    output_list = []
    error_count = 0
    raw_needed_keys = ["customer_id", "history", "target"]
    
    with open(os.path.join(raw_data_path, raw_data_name), 'r', encoding='utf8') as f:
        for line in f:
            raw_data = json.loads(line)
            tmp_dict = {k: raw_data[k] for k in raw_needed_keys}
            
            if isinstance(random_seed, int):
                tmp_dict["customer_id"] = tmp_dict["customer_id"]
                
            tmp_dict["prompt"] = raw_data["request"]["contents"][0]["parts"][0]["text"]
            
            if tmp_dict["customer_id"] not in responses_dict:
                logging.warning(f"Missing response for customer_id: {tmp_dict['customer_id']}")
                error_count += 1
                continue
                
            tmp_dict["response"] = responses_dict[tmp_dict["customer_id"]]
            output_list.append(tmp_dict)
    
    logging.info(f"Total raw data processing errors: {error_count}")
    return output_list

def sort_key(item: Dict) -> Tuple[str, str, int]:
    customer_id = item['customer_id']
    first_part, middle_part, last_part = customer_id.split('_')
    return (middle_part, last_part, int(first_part))

def main():
    setup_logging()
    args = parse_args()
    
    raw_data_path, generated_data_path, output_path = setup_paths(args)
    generated_files = os.listdir(generated_data_path)
    print(f"generated_files: {generated_files}")
    
    raw_data_name, generated_data_name = get_datanames(
        args.dataset, args.data_split, generated_files
    )
    
    logging.info(f"Processing files: {raw_data_name} -> {generated_data_name}")
    
    responses_dict = load_responses(generated_data_path, generated_data_name)
    output_list = process_raw_data(
        raw_data_path, raw_data_name, responses_dict, args.data_split
    )
    
    output_list.sort(key=sort_key)
    output_file = os.path.join(output_path, raw_data_name)
    
    with open(output_file, 'w', encoding="utf8") as f:
        for item in output_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logging.info(f"Successfully processed {len(output_list)} items")
    logging.info(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()