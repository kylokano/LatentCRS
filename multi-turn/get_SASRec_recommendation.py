from omegaconf import OmegaConf
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

from models import PriorNetwork
from utils import PrecalculatedRecModelConfig, recall_at_k, ndcg_k
from pre_rec_model.rec_models_cp import SASRecModel
from mydatasets import MStepDataset, collect_fn_for_msstep, get_useritem_rawid2modelid

raw_models_file_path = "./"

def load_model(dataset_name, config_path, generative_model_path):
    cfg = OmegaConf.load(config_path+"/config.yaml")
    
    print("Loading Intention Embedding(KMeans)...")
    centers = np.load(os.path.join(raw_models_file_path, cfg.datasets.precalculated_recmodel.kmeans_file_name, "centers.npy"))
    centers = torch.tensor(centers)
    print(f"Precalculated KMeans centers shape: {centers.shape}")
    
    cfg.intent_hidden = centers.shape[1]
    
    print(f"Loading precalculated recommendation model from {cfg.datasets.precalculated_recmodel.checkpoint}")
    precalc_recmodel_config = json.load(open(os.path.join(raw_models_file_path, cfg.datasets.precalculated_recmodel.config)))
    precalc_recmodel_config = PrecalculatedRecModelConfig(precalc_recmodel_config)
    precalc_recmodel = SASRecModel(precalc_recmodel_config)
    precalc_recmodel.load_state_dict(torch.load(os.path.join(raw_models_file_path, cfg.datasets.precalculated_recmodel.checkpoint), weights_only=True))
    print("Precalculated recommendation model loaded.")
    
    print("Getting user and item rawid to modelid mapping...")
    id2modelid = get_useritem_rawid2modelid(os.path.join(raw_models_file_path, cfg.datasets.raw_data_path))
    print("Apply the mapping to precalculated recmodel...")
    item_modelid2rawid = id2modelid["item_modelid2rawid"]
    item_modelid2rawid[0] = 0
    item_modelid2rawid = list(map(int, item_modelid2rawid))
    if len(item_modelid2rawid) != precalc_recmodel.item_embeddings.weight.data.shape[0]:
        print(f"max raw item idx: {max(item_modelid2rawid)}")
        print(f"length of new_idx: {len(item_modelid2rawid)}")
        print(f"length of item embeddings: {precalc_recmodel.item_embeddings.weight.data.shape[0]}")
        print(f"Warning: length of new_idx does not match the length of item embeddings, please check the dataset to ensure the feature.")
    item_modelid2rawid = torch.tensor(item_modelid2rawid)
    precalc_recmodel.item_embeddings.weight.data = precalc_recmodel.item_embeddings.weight.data[item_modelid2rawid]
    
    cfg.precalculated_recmodel.item_size = len(item_modelid2rawid)
    
    prior_model = PriorNetwork(cfg, centers, precalc_recmodel)
    cfg.precalculated_recmodel.item_size = len(item_modelid2rawid)
    item_size = cfg.precalculated_recmodel.item_size
    return id2modelid, prior_model, item_size

def load_dataset(dataset_name, config_path, dataset_path, precal_embeddinng_path, id2modelid, item_size):
    cfg = OmegaConf.load(config_path+"/config.yaml")
    cfg.precalculated_recmodel.item_size = item_size
    test_dataset = MStepDataset(
        data_path=dataset_path,
        precalculated_embeddings_path=precal_embeddinng_path,
        raw_data_path=os.path.join(raw_models_file_path, cfg.datasets.raw_data_path),
        meta_task_keys=cfg.datasets.meta_task_keys,
        filter_keys=cfg.datasets.filter_keys,
        neg_sample_size=cfg.neg_sample_size,
        id2modelid=id2modelid,
        data_type="test",
        is_debug=cfg.debug
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        collate_fn=collect_fn_for_msstep,
        num_workers=4,
        shuffle=False,
        sampler=SequentialSampler(test_dataset)
    )
    print("Dataset loaded.")
    return test_loader
    
def get_rs_predict(prior_model, test_loader):
    num_items = prior_model.item_embedding.weight.shape[0]
    # get predictions
    prior_model.eval()
    prior_model.to("cuda")
    for batch_idx, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Full-sort predict"):
        test_batch = [value.to("cuda") for value in test_batch]
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        truely_item = target_item.cpu().data.numpy().copy()
        truely_item = [[i] for i in truely_item]
        
        target_item = torch.arange(num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
        p_y = prior_model.get_user_seq_predict(histories, history_mask, target_item)
        scores = p_y
        
        scores = scores.detach().cpu().data.numpy().copy()
        # get top 20 items
        ind_without_text = np.argpartition(scores, -50)[:, -50:]
        arr_ind_without_text = scores[np.arange(len(scores))[:, None], ind_without_text]
        arr_ind_argsort_without_text = np.argsort(arr_ind_without_text)[:, ::-1]
        batch_pred_list_without_text = ind_without_text[np.arange(len(scores))[:, None], arr_ind_argsort_without_text]
        top50_item = batch_pred_list_without_text
        
        if batch_idx == 0:
            logits_pred_list = top50_item
            truely_item_list = truely_item
        else:
            logits_pred_list = np.append(logits_pred_list, top50_item, axis=0)
            truely_item_list = np.append(truely_item_list, truely_item, axis=0)
    
    return logits_pred_list, truely_item_list

def get_raw_itemid(pred_list, model_id2raw_id_dict):
    output_list = []
    for per_line in pred_list:
        output_list.append([model_id2raw_id_dict[int(item)] for item in per_line])
    return output_list

def get_recommendation(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, prior_model, item_size, given_topK=5, item_features_constraint=None, already_recommend=None):
    test_loader = load_dataset(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, item_size)
    logits_pred_list, truely_item_list = get_rs_predict(prior_model, test_loader)
    pred_itemids = logits_pred_list
    get_full_sort_score(0, "test_raw", truely_item_list, logits_pred_list)
    
    pred_raw_id, target_raw_id = [], []
    for idx in range(len(truely_item_list)):
        pred_raw_id.append([id2modelid["item_modelid2rawid"][int(item)] for item in pred_itemids[idx]])
        target_raw_id.append([id2modelid["item_modelid2rawid"][int(item)] for item in truely_item_list[idx]])
    
    item_title_dict, item_features_dict = {}, {}
    if dataset_name == "ml-1m":
        item_features_path = "./data/ml-1m/movie_features.csv"
        item_features_file = pd.read_csv(item_features_path, sep="\t")
        for pre_ids, title, features in zip(item_features_file["MovieID"], item_features_file["Title"], item_features_file["Genres"]):
            item_title_dict[str(pre_ids)] = title
            item_features_dict[str(pre_ids)] = features.split("|")
    elif dataset_name == "VedioGames":
        item_features_path = "./data/VedioGames/VedioGames_features.csv"
        item_features_file = pd.read_csv(item_features_path, sep="\t")
        for pre_ids, title, features in zip(item_features_file["GamesID"], item_features_file["title"], item_features_file["categories"]):
            item_title_dict[str(pre_ids)] = title
            if type(features) != str:   
                features = "test, test"
            item_features = features.split(",")
            item_features_dict[str(pre_ids)] = [item.strip() for item in item_features if item!="Video Games"]
    elif dataset_name == "cds":
        item_features_path = "./data/cds/CDs_features.csv"
        item_features_file = pd.read_csv(item_features_path, sep="\t")
        for pre_ids, title, features in zip(item_features_file["CDsID"], item_features_file["title"], item_features_file["categories"]):
            item_title_dict[str(pre_ids)] = title
            item_features = features.split(",")
            item_features_dict[str(pre_ids)] = [item.strip() for item in item_features if item!="CDs & Vinyl"]
    
    
    recommend_items, recommend_item_features, recommend_item_ids = [], [], []
    for item_id in pred_raw_id:
        recommend_items.append([item_title_dict[str(item)] for item in item_id if str(item) in item_title_dict])
        recommend_item_features.append([item_features_dict[str(item)] for item in item_id if str(item) in item_features_dict])
        recommend_item_ids.append([item for item in item_id if str(item) in item_title_dict])
    
    get_full_sort_score(0, "test_raw", target_raw_id, recommend_item_ids)
    
    return recommend_item_ids, recommend_items, recommend_item_features # 推荐的ID，推荐的item title，推荐的item features


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def get_full_sort_score(epoch, prefix, answers, pred_list):
    recall, ndcg = [], []
    for k in [1, 5]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "HIT@1": recall[0],
        "NDCG@1": ndcg[0],
        "HIT@5": recall[1],
        "NDCG@5": ndcg[1],
        # "HIT@15": recall[2],
        # "NDCG@15": ndcg[2],
        # "HIT@20": recall[3],
        # "NDCG@20": ndcg[3],
    }
    print(epoch, prefix, post_fix)
    return post_fix
    
if __name__ == "__main__":
    # test the results
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    config_path = "games_em_allpretrain_0.001_m_rec_loss1_intent_num128_m_negetive_loss1"
    dataset_name = "VedioGames"
    
    config_path = "ml_em_allpretrain_0.001_m_rec_loss5_intent_num512_m_negetive_loss0"
    dataset_name = "ml-1m"
    
    config_path = "cds_generative0.005_m_rec_loss10_intent_num128"
    dataset_name = "cds"
    
    generative_model_path = ""
    dataset_path = f"./results/ours-llama3/{dataset_name}_raw_seedtest_sample_500.jsonl"
    precal_embedding_path = f"./text2embedding/embedding_cache/{dataset_name}"  
    id2modelid, prior_model, item_size = load_model(dataset_name, config_path, generative_model_path)
    recommend_item_ids, recommend_items, recommend_item_features = get_recommendation(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, prior_model, item_size)