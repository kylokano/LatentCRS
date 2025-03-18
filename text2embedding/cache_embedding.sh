dataset_name=ml-1m # ml-1m, cds, VedioGames
model_path=llama-3.1-8b-instruct
batch_size=1
gpu_id=3

python3 text2embedding/text_embedding_cache.py --dataset $dataset_name --model_path $model_path --batch_size $batch_size --gpu_id $gpu_id
