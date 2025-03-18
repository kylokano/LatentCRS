import os
from utils import load_clusters
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cds")
    parser.add_argument("--num_cluster", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=5)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    num_cluster = args.num_cluster
    gpu_id = args.gpu_id

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    file_path = f"pre_rec_model/output/{dataset_name}/"
    cluster_file_path = [i for i in os.listdir(file_path) if "clusters" in i and f"{num_cluster}" in i]
    for cluster_file in cluster_file_path:
        precalc_kmeans_model = load_clusters(os.path.join(file_path, cluster_file))
        centers = precalc_kmeans_model[0].centroids.cpu().numpy()
        np.save(os.path.join(file_path, cluster_file, "centers.npy"), centers)
        print(f"{cluster_file} centers: {centers.shape}")

