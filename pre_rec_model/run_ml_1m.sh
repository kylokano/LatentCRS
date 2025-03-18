num_intent=(128 256 512 1024)
contrast_type=(Hybrid)
gpu_id=1

for n_int in "${num_intent[@]}"; do
    for ct in "${contrast_type[@]}"; do
        python3 main.py --data_name ml-1m --cf_weight 0.1 \
        --model_idx num_cluster_"${n_int}"_"${ct}" --gpu_id "${gpu_id}" \
        --batch_size 256 --contrast_type "${ct}" \
        --num_intent_cluster "${n_int}" --seq_representation_type mean \
        --warm_up_epoches 0 --intent_cf_weight 0.1 --num_hidden_layers 2 \
        --max_seq_length 200 --model_name IntentRec --n_views 3
    done
    python3 get_centers.py --dataset_name ml-1m --num_cluster "${n_int}" --gpu_id "${gpu_id}"
done
