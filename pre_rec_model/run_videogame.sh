num_intent=(128 256 512 1024)
contrast_type=(Hybrid)
gpu_id=1

for n_int in "${num_intent[@]}"; do
    for ct in "${contrast_type[@]}"; do
        python3 main.py --data_name VedioGames --cf_weight 0.1 \
        --model_idx num_cluster_"${n_int}"_"${ct}" --gpu_id "${gpu_id}" \
        --batch_size 256 --contrast_type "${ct}" \
        --num_intent_cluster "${n_int}" --seq_representation_type mean \
        --warm_up_epoches 0 --intent_cf_weight 0.1 --num_hidden_layers 3
    done
    python3 get_centers.py --dataset_name VedioGames --num_cluster "${n_int}" --gpu_id "${gpu_id}"
done
