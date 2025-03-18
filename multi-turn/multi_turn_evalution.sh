export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
dataset_name=ml-1m
rec_model_config_path=<path_to_your_model_config_saved_path>
rec_model_path=<path_to_your_saved_prior_model>
test_path=<path_to_your_sampled_test_data>
given_topK=5 # how many topK items are given to the user similaritor
python multi_turn.py --from_json ${test_path} --dataset_name ${dataset_name} --recommend_model_config_path $rec_model_config_path --recommend_model_path $rec_model_path --given_topK $given_topK


