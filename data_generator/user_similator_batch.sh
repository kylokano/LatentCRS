# ml-1m, cds, VedioGames
dataset_name=ml-1m 

google_cloud_credentials="<your google cloud credentials path>"
google_cloud_project="<your google cloud project>"
google_cloud_location="<your google cloud location>"
save_bucket="<your google cloud bucket>"

python3 data_generator/batch_data_simulator.py --dataset "${dataset_name}"
python3 data_generator/combination_prompt.py --dataset "${dataset_name}"

# run batch api for train set
python3 batch_api.py --input_file_path "${dataset_name}_combination.jsonl" --save_bucket "${save_bucket}" --google_cloud_credentials "${google_cloud_credentials}" --google_cloud_project "${google_cloud_project}" --google_cloud_location "${google_cloud_location}"
# run batch api for valid set
python3 batch_api.py --input_file_path "${dataset_name}_valid.jsonl" --save_bucket "${save_bucket}" --google_cloud_credentials "${google_cloud_credentials}" --google_cloud_project "${google_cloud_project}" --google_cloud_location "${google_cloud_location}"
# run batch api for test set
python3 batch_api.py --input_file_path "${dataset_name}_test.jsonl" --save_bucket "${save_bucket}" --google_cloud_credentials "${google_cloud_credentials}" --google_cloud_project "${google_cloud_project}" --google_cloud_location "${google_cloud_location}"

# process generated data
python3 data_generator/process_generated_data.py --dataset "${dataset_name}" --data_split combination
python3 data_generator/process_generated_data.py --dataset "${dataset_name}" --data_split valid
python3 data_generator/process_generated_data.py --dataset "${dataset_name}" --data_split test


# sample 500 users for multi-turn
python3 data_generator/simple4multi_turn.py