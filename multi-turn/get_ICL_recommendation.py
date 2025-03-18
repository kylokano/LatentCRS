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

def load_model(dataset_name, config_path, generative_model_path):
    cfg = OmegaConf.load(config_path+"/config.yaml")
    
    print("Loading Intention Embedding(KMeans)...")
    centers = np.load(os.path.join(cfg.datasets.precalculated_recmodel.kmeans_file_name, "centers.npy"))
    centers = torch.tensor(centers)
    print(f"Precalculated KMeans centers shape: {centers.shape}")
    
    cfg.intent_hidden = centers.shape[1]
    
    print(f"Loading precalculated recommendation model from {cfg.datasets.precalculated_recmodel.checkpoint}")
    precalc_recmodel_config = json.load(open(cfg.datasets.precalculated_recmodel.config))
    precalc_recmodel_config = PrecalculatedRecModelConfig(precalc_recmodel_config)
    precalc_recmodel = SASRecModel(precalc_recmodel_config)
    precalc_recmodel.load_state_dict(torch.load(cfg.datasets.precalculated_recmodel.checkpoint, weights_only=True))
    print("Precalculated recommendation model loaded.")
    
    print("Getting user and item rawid to modelid mapping...")
    id2modelid = get_useritem_rawid2modelid(cfg.datasets.raw_data_path)
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
    print("Loading trained weights...")
    pretrained_model_weights = torch.load(generative_model_path, weights_only=True)
    prior_model.load_state_dict(pretrained_model_weights)
    print("Trained weights loaded.")
    cfg.precalculated_recmodel.item_size = len(item_modelid2rawid)
    item_size = cfg.precalculated_recmodel.item_size
    return id2modelid, prior_model, item_size

def load_dataset(dataset_name, config_path, dataset_path, precal_embeddinng_path, id2modelid, item_size):
    cfg = OmegaConf.load(config_path+"/config.yaml")
    cfg.precalculated_recmodel.item_size = item_size
    test_dataset = MStepDataset(
        data_path=dataset_path,
        precalculated_embeddings_path=precal_embeddinng_path,
        raw_data_path=cfg.datasets.raw_data_path,
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
        p_m_ux = prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = prior_model.get_y_m(target_item, intent_distribution=p_m_ux) # [batch_size, num_items]
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

def load_model_and_dataset(dataset_name, config_path, generative_model_path, dataset_path, precal_embeddinng_path):
    cfg = OmegaConf.load(config_path+"/config.yaml")
    
    print("Loading Intention Embedding(KMeans)...")
    centers = np.load(os.path.join(cfg.datasets.precalculated_recmodel.kmeans_file_name, "centers.npy"))
    centers = torch.tensor(centers)
    print(f"Precalculated KMeans centers shape: {centers.shape}")
    
    cfg.intent_hidden = centers.shape[1]
    
    print(f"Loading precalculated recommendation model from {cfg.datasets.precalculated_recmodel.checkpoint}")
    precalc_recmodel_config = json.load(open(cfg.datasets.precalculated_recmodel.config))
    precalc_recmodel_config = PrecalculatedRecModelConfig(precalc_recmodel_config)
    precalc_recmodel = SASRecModel(precalc_recmodel_config)
    precalc_recmodel.load_state_dict(torch.load(cfg.datasets.precalculated_recmodel.checkpoint, weights_only=True))
    print("Precalculated recommendation model loaded.")
    
    print("Getting user and item rawid to modelid mapping...")
    id2modelid = get_useritem_rawid2modelid(cfg.datasets.raw_data_path)
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

    test_dataset = MStepDataset(
        data_path=dataset_path,
        precalculated_embeddings_path=precal_embeddinng_path,
        raw_data_path=cfg.datasets.raw_data_path,
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
    
    prior_model = PriorNetwork(cfg, centers, precalc_recmodel)
    print("Loading trained weights...")
    pretrained_model_weights = torch.load(generative_model_path, weights_only=True)
    prior_model.load_state_dict(pretrained_model_weights)
    print("Trained weights loaded.")
    
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
        p_m_ux = prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = prior_model.get_y_m(target_item, intent_distribution=p_m_ux) # [batch_size, num_items]
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


def get_recommendation_without_filter(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, prior_model, item_size, given_topK=5, item_features_constraint=None, already_recommend=None):
    test_loader = load_dataset(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, item_size)
    logits_pred_list, truely_item_list = get_rs_predict(prior_model, test_loader)
    pred_itemids = logits_pred_list
    
    
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
    is_success_raw = []
    for each_line, target_items in zip(recommend_item_ids, target_raw_id):
        if target_items[0] in each_line[:given_topK]:
            is_success_raw.append(1)
        else:
            is_success_raw.append(0)
    print("raw_success_rate", sum(is_success_raw)/len(is_success_raw))
    
    
    if item_features_constraint is None:
        filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features = pred_itemids, recommend_items, recommend_item_features
    else:
        filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features = [], [], []
        for pred_itemid, recommend_item, recommend_item_feature, per_item_features_constraint, already_recommend_item in zip(recommend_item_ids, recommend_items, recommend_item_features, item_features_constraint, already_recommend):
            # 首先基于already_recommend进行过滤和重排序
            if len(already_recommend_item) != 0:
                already_rec_set = set(already_recommend_item)
                tmp_items_with_scores = []
                
                for i in range(len(pred_itemid)):
                    tmp_items_with_scores.append({
                        'pred_id': pred_itemid[i],
                        'recommend_item': recommend_item[i],
                        'item_features': recommend_item_feature[i],
                        'in_already_recommend': pred_itemid[i] in already_rec_set,
                        'original_idx': i
                    })
                
                # 将已推荐过的项目排在后面
                sorted_items = sorted(tmp_items_with_scores, 
                                   key=lambda x: (x['in_already_recommend'], x['original_idx']))
                
                # 重新组织排序后的列表
                pred_itemid = [item['pred_id'] for item in sorted_items]
                recommend_item = [item['recommend_item'] for item in sorted_items]
                recommend_item_feature = [item['item_features'] for item in sorted_items]
                
                filtered_pred_itemids.append(pred_itemid)
                filtered_recommend_items.append(recommend_item)
                filtered_recommend_item_features.append(recommend_item_feature)
                # filtered_pred_itemids.append(tmp_pred_itemids[:given_topK])
                # filtered_recommend_items.append(tmp_recommend_items[:given_topK])
                # filtered_recommend_item_features.append(tmp_recommend_item_features[:given_topK])
        get_full_sort_score(0, "test_filtered", target_raw_id, filtered_pred_itemids)
        filtered_pred_itemids = [item[:given_topK] for item in filtered_pred_itemids]
        filtered_recommend_items = [item[:given_topK] for item in filtered_recommend_items]
        filtered_recommend_item_features = [item[:given_topK] for item in filtered_recommend_item_features]
        
        is_success = []
        for each_line, target_items in zip(filtered_pred_itemids, target_raw_id):
            if len(each_line) != given_topK:
                print("each_line", each_line)
            if target_items[0] in each_line:
                is_success.append(1)
            else:
                is_success.append(0)
    return filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features, is_success

def get_recommendation(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, prior_model, item_size, given_topK=5, item_features_constraint=None, already_recommend=None):
    test_loader = load_dataset(dataset_name, config_path, dataset_path, precal_embedding_path, id2modelid, item_size)
    logits_pred_list, truely_item_list = get_rs_predict(prior_model, test_loader)
    pred_itemids = logits_pred_list
    
    if already_recommend is None:
        already_recommend = []
        for _ in range(len(pred_itemids)):
            already_recommend.append([])
    
    
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
            if type(features) == str:
                item_features = features.split(",")
            else:
                item_features = []
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
    is_success_raw = []
    for each_line, target_items in zip(recommend_item_ids, target_raw_id):
        if target_items[0] in each_line[:given_topK]:
            is_success_raw.append(1)
        else:
            is_success_raw.append(0)
    print("raw_success_rate", sum(is_success_raw)/len(is_success_raw))
    
    
    if item_features_constraint is None:
        filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features = pred_itemids, recommend_items, recommend_item_features
    else:
        filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features = [], [], []
        for pred_itemid, recommend_item, recommend_item_feature, per_item_features_constraint, already_recommend_item in zip(recommend_item_ids, recommend_items, recommend_item_features, item_features_constraint, already_recommend):
            # 首先基于already_recommend进行过滤和重排序
            if len(already_recommend_item) != 0:
                already_rec_set = set(already_recommend_item)
                tmp_items_with_scores = []
                
                for i in range(len(pred_itemid)):
                    tmp_items_with_scores.append({
                        'pred_id': pred_itemid[i],
                        'recommend_item': recommend_item[i],
                        'item_features': recommend_item_feature[i],
                        'in_already_recommend': pred_itemid[i] in already_rec_set,
                        'original_idx': i
                    })
                
                # 将已推荐过的项目排在后面
                sorted_items = sorted(tmp_items_with_scores, 
                                   key=lambda x: (x['in_already_recommend'], x['original_idx']))
                
                # 重新组织排序后的列表
                pred_itemid = [item['pred_id'] for item in sorted_items]
                recommend_item = [item['recommend_item'] for item in sorted_items]
                recommend_item_feature = [item['item_features'] for item in sorted_items]
            
            # 继续原有的特征约束过滤逻辑
            if len(item_features_constraint) == 0:
                filtered_pred_itemids.append(pred_itemid)
                filtered_recommend_items.append(recommend_item)
                filtered_recommend_item_features.append(recommend_item_feature)
            else:
                tmp_items_with_scores = []
                for idx in range(len(pred_itemid)):
                    per_item_features_constraint = [item.strip().lower() for item in per_item_features_constraint]
                    recommend_item_feature[idx] = [item.strip().lower() for item in recommend_item_feature[idx]]
                    # 计算特征交集的大小，并保存原始索引
                    intersection_size = len(set(per_item_features_constraint) & set(recommend_item_feature[idx]))
                    tmp_items_with_scores.append({
                        'pred_id': pred_itemid[idx],
                        'recommend_item': recommend_item[idx],
                        'item_features': recommend_item_feature[idx],
                        'intersection_size': intersection_size,
                        'original_idx': idx  # 保存原始索引
                    })
                
                # 按照交集大小从大到小排序，相同时按照原始索引排序
                sorted_items = sorted(tmp_items_with_scores, 
                                   key=lambda x: (-x['intersection_size'], x['original_idx']))
                
                # 重新组织排序后的列表
                tmp_pred_itemids = [item['pred_id'] for item in sorted_items]
                tmp_recommend_items = [item['recommend_item'] for item in sorted_items]
                tmp_recommend_item_features = [item['item_features'] for item in sorted_items]
                
                filtered_pred_itemids.append(tmp_pred_itemids)
                filtered_recommend_items.append(tmp_recommend_items)
                filtered_recommend_item_features.append(tmp_recommend_item_features)
                # filtered_pred_itemids.append(tmp_pred_itemids[:given_topK])
                # filtered_recommend_items.append(tmp_recommend_items[:given_topK])
                # filtered_recommend_item_features.append(tmp_recommend_item_features[:given_topK])
        get_full_sort_score(0, "test_filtered", target_raw_id, filtered_pred_itemids)
        filtered_pred_itemids = [item[:given_topK] for item in filtered_pred_itemids]
        filtered_recommend_items = [item[:given_topK] for item in filtered_recommend_items]
        filtered_recommend_item_features = [item[:given_topK] for item in filtered_recommend_item_features]
        
        is_success = []
        for each_line, target_items in zip(filtered_pred_itemids, target_raw_id):
            if len(each_line) != given_topK:
                print("each_line", each_line)
            if target_items[0] in each_line:
                is_success.append(1)
            else:
                is_success.append(0)
        # print(sum(is_success_raw)/len(is_success_raw))
        # print(sum(is_success)/len(is_success))
        
        raise_success_count, down_success_count = 0, 0
        for idx_in in range(len(is_success_raw)):
            # if is_success_raw[idx_in] == 1 and is_success[idx_in] == 0:
            #     print("--------------------------------")
            #     print("idx_in", idx_in)
            #     print("item_features_constraint", item_features_constraint[idx_in])
            #     print("target_item", truely_item_list[idx_in])
            #     print("raw_id_target", target_raw_id[idx_in])
            #     print("recommend", recommend_item_ids[idx_in])
            #     print("recommend_item_feature", recommend_item_features[idx_in])
            #     print("filtered_recommend", filtered_pred_itemids[idx_in])
            #     print("--------------------------------")
            if is_success_raw[idx_in] == 0 and is_success[idx_in] == 1:
                raise_success_count += 1
            if is_success_raw[idx_in] == 1 and is_success[idx_in] == 0:
                down_success_count += 1
        print("raise_success_count", raise_success_count)
        print("down_success_count", down_success_count)
    
    return filtered_pred_itemids, filtered_recommend_items, filtered_recommend_item_features, is_success


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
    for k in [1, 5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "HIT@1": recall[0],
        "NDCG@1": ndcg[0],
        "HIT@5": recall[1],
        "NDCG@5": ndcg[1],
        "HIT@10": recall[2],
        "NDCG@10": ndcg[2],
        "HIT@15": recall[3],
        "NDCG@15": ndcg[3],
        "HIT@20": recall[4],
        "NDCG@20": ndcg[4],
    }
    print(epoch, prefix, post_fix)
    return post_fix
    
if __name__ == "__main__":
    # test the results