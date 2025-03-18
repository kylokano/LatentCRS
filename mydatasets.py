from typing import Set
from collections import OrderedDict
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import json
import os
import random
import torch
import numpy as np
from utils import rank0_print, set_seed, get_local_run_dir
import pandas as pd
import pickle

def get_useritem_rawid2modelid(raw_data_path: str):
    """
    Convert raw user and item IDs to model IDs and create mapping dictionaries.
    
    Args:
        raw_data_path: Path to raw data file containing user-item interactions
        
    Returns:
        Dictionary containing:
        - user_rawid2modelid: Mapping from raw user ID to model ID
        - item_rawid2modelid: Mapping from raw item ID to model ID  
        - user_modelid2rawid: Mapping from model ID to raw user ID
        - item_modelid2rawid: Mapping from model ID to raw item ID
    """
    user_set, item_set = OrderedDict(), OrderedDict()
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            user_id, item_list = line.strip().split("\t")
            item_list = item_list.split(" ")
            if user_id not in user_set:
                user_set[user_id] = 1
            for item in item_list:
                if item not in item_set:
                    item_set[item] = 1
    rank0_print(f"Total number of users: {len(user_set)}")
    rank0_print(f"Total number of items: {len(item_set)}")
    
    user_rawid2modelid, user_modelid2rawid = {"[PAD]": 0}, ["PAD"]
    for per_user in user_set:
        user_rawid2modelid[per_user] = len(user_rawid2modelid)
        user_modelid2rawid.append(per_user)
    item_rawid2modelid, item_modelid2rawid = {"[PAD]": 0}, ["PAD"]
    for per_item in item_set:
        item_rawid2modelid[per_item] = len(item_rawid2modelid)
        item_modelid2rawid.append(per_item)
    assert len(user_modelid2rawid) == len(user_rawid2modelid), "The length of user_modelid2rawid and user_rawid2modelid should be the same"
    assert len(item_modelid2rawid) == len(item_rawid2modelid), "The length of item_modelid2rawid and item_rawid2modelid should be the same"
    assert len(user_modelid2rawid) == len(user_set) + 1, "The length of user_modelid2rawid should be the same as the number of users plus 1"
    assert len(item_modelid2rawid) == len(item_set) + 1, "The length of item_modelid2rawid should be the same as the number of items plus 1"
    
    return {"user_rawid2modelid": user_rawid2modelid, "item_rawid2modelid": item_rawid2modelid, "user_modelid2rawid": user_modelid2rawid, "item_modelid2rawid": item_modelid2rawid}


class MStepDataset(Dataset):
    """
    Dataset class for training model from EM training data.
    Only used for training and validation, not for testing.
    
    Args:
        data_path: Path to interaction data
        precalculated_embeddings_path: Path to precalculated embeddings
        raw_data_path: Path to raw data for negative sampling
        data_type: Type of data ('train', 'valid' or 'test')
        meta_task_keys: List of meta task keys
        filter_keys: List of keys to filter data
        neg_sample_size: Number of negative samples per positive sample
        id2modelid: Dictionary mapping raw IDs to model IDs
        is_debug: Whether in debug mode
    """
    def __init__(self, data_path: str, precalculated_embeddings_path: str, raw_data_path: str, data_type: str, meta_task_keys: list[str], filter_keys: list[str], neg_sample_size: int, id2modelid: dict, is_debug: bool = False):
        self.data_path = data_path
        self.precalculated_embeddings_path = precalculated_embeddings_path
        # Load raw data for negative sampling
        self.raw_data_path = raw_data_path
        self.data_type = data_type
        assert self.data_type in ["train", "valid", "test"], "data_type should be train or valid or test"
        
        self.meta_task_keys = meta_task_keys
        print(f"meta_task_keys: {self.meta_task_keys}")
        self.filter_keys = filter_keys if data_type == "train" else [self.data_type]
        self.neg_sample_size = neg_sample_size
        
        self.user_id, self.customer_id, self.history, self.target_item, self.precalculated_embeddings = self.load_data()
        print("Data loaded")
    
        if is_debug:
            # Truncate data in debug mode
            self.user_id = self.user_id[:20]
            self.customer_id = self.customer_id[:20]
            self.history = self.history[:20]
            self.target_item = self.target_item[:20]
            self.precalculated_embeddings = self.precalculated_embeddings[:20]
    
        self.user_rawid2modelid = id2modelid["user_rawid2modelid"]
        self.item_rawid2modelid = id2modelid["item_rawid2modelid"]
        self.user_modelid2rawid = id2modelid["user_modelid2rawid"]
        self.item_modelid2rawid = id2modelid["item_modelid2rawid"]
        
        self.user_id_set = set(self.user_rawid2modelid.values())
        self.item_id_set = set(self.item_rawid2modelid.values())
        
        # Convert raw IDs to model IDs
        self.user_id = list(map(lambda x: self.user_rawid2modelid[x], self.user_id))
        self.history = [list(map(lambda x: self.item_rawid2modelid[x], h)) for h in self.history]
        self.target_item = list(map(lambda x: self.item_rawid2modelid[x], self.target_item))
        
        # Build user interaction dictionary
        self.user_interaction_dict = {}
        for user_id, history in zip(self.user_id, self.history):
            self.user_interaction_dict[user_id] = history
        neg_cache_path = data_path + ".neg_cache"
        if os.path.exists(neg_cache_path):
            print("Loading cached negative items from ", neg_cache_path)
            with open(neg_cache_path, 'rb') as f:
                self.neg_items = pickle.load(f)
        else:
            self.get_neg_items()
            print("Saving cached negative items to ", neg_cache_path)
            with open(neg_cache_path, 'wb') as f:
                pickle.dump(self.neg_items, f)

    def neg_sample(self, user_id: int, target_item: int):
        """
        Sample negative items for a user-item pair
        
        Args:
            user_id: User ID
            target_item: Target item ID
            
        Returns:
            List of negative item IDs
        """
        neg_items = []
        user_items = set(self.user_interaction_dict[user_id])
        candidate_items = list(self.item_id_set - user_items)  # Get items not interacted by user
        
        # Random sample enough negative samples at once, then filter out target item
        neg_candidates = random.sample(candidate_items, min(self.neg_sample_size + 1, len(candidate_items)))
        neg_items = [item for item in neg_candidates if item != target_item and item != 0][:self.neg_sample_size]
        # Continue sampling if not enough negative samples
        if len(neg_items) < self.neg_sample_size:
            while len(neg_items) < self.neg_sample_size:
                item = random.choice(candidate_items) 
                if item != target_item and item != 0 and item not in neg_items:
                    neg_items.append(item)
        return neg_items
    
    def get_neg_items(self):
        """Get negative items for all user-item pairs"""
        print("Getting negative items")
        if self.neg_sample_size > 0:
            self.neg_items = []
            for user_id, target_item in zip(self.user_id, self.target_item):
                neg_items = self.neg_sample(user_id, target_item)
                self.neg_items.append(neg_items)
        else:
            self.neg_items = [0] * len(self.all_users) # dummy neg_items
        assert len(self.neg_items) == len(self.user_id), "The length of neg_items and user_id should be the same"
        print("Negative items got")
        
    def load_data(self):
        """
        Load interaction data and precalculated embeddings
        
        Returns:
            user_id: List of user IDs
            customer_id: List of customer IDs
            history: List of interaction histories
            target_item: List of target items
            precalculated_embeddings: List of precalculated embeddings
        """
        user_id, customer_id, history, target_item, precalculated_embeddings = [], [], [], [], []
        filter_key_cache = None
        rank0_print(f"Loading data from {self.data_path}")
        rank0_print(f"Filter keys: {self.filter_keys}")
        for filter_key in self.filter_keys:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line) == 0:
                        continue
                    data = json.loads(line)
                    filter_key = data["customer_id"].split("_")[-1]
                    if filter_key not in self.filter_keys:
                        continue
                    if filter_key != filter_key_cache:
                        print(f"Loading precalculated embeddings for {filter_key}")
                        filter_key_cache = filter_key
                        customerid2embidx = self.load_precalculated_embeddings(filter_key)
                        print(f"Loaded precalculated embeddings for {filter_key}")

                    user_id.append(data["customer_id"].split("_")[0])
                    customer_id.append(data["customer_id"])
                    history.append(data["history"])
                    target_item.append(data["target"])
                    emb_idx = customerid2embidx[data["customer_id"]]
                    precalculated_embeddings.append(emb_idx)
        assert len(user_id) == len(precalculated_embeddings), "The length of user_id and precalculated_embeddings should be the same"
        return user_id, customer_id, history, target_item, precalculated_embeddings
                
    def load_precalculated_embeddings(self, filter_key: str):
        """
        Load precalculated embeddings for a filter key
        
        Args:
            filter_key: Filter key to load embeddings for
            
        Returns:
            Dictionary mapping user IDs to embeddings
        """
        userid2embeddings = {}
        for meta_task_key in self.meta_task_keys:
            file_suffix = self.data_type if self.data_type != "train" else filter_key
            if self.precalculated_embeddings_path.endswith(".pt"):
                precalculated_embeddings_file = self.precalculated_embeddings_path
            else:
                precalculated_embeddings_file = os.path.join(self.precalculated_embeddings_path, f"{meta_task_key}_{file_suffix}.pt")
            precalculated_data = torch.load(precalculated_embeddings_file, weights_only=True)
            special_id_list, embeddings = precalculated_data["special_id"], precalculated_data["embeddings"].float()
            for idx, special_id in enumerate(special_id_list):
                if special_id not in userid2embeddings:
                    userid2embeddings[special_id] = {}
                userid2embeddings[special_id][meta_task_key] = embeddings[idx]
        return userid2embeddings

    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self, idx):
        return self.user_id[idx], self.history[idx], self.target_item[idx], self.neg_items[idx], self.precalculated_embeddings[idx]
    
    

def collect_fn_for_msstep(batch):
    """
    Collate function for EM training dataset
    
    Args:
        batch: List of tuples containing (user_id, history, target_item, neg_items, precalculated_embeddings)
        
    Returns:
        Tuple of tensors:
        - user_id: User IDs
        - history: Padded interaction histories
        - history_mask: Attention mask for histories
        - target_item: Target items
        - pos_neg_items: Concatenated positive and negative items
        - precalc_emb: Mean of precalculated embeddings
        - labels: Labels for positive/negative items
    """
    user_id, history, target_item, neg_items, precalculated_embeddings = zip(*batch)
    meta_task_keys = list(precalculated_embeddings[0].keys())
    
    # Create dictionary to store tensors for each meta task
    precalc_emb_dict = {}
    for meta_task_key in meta_task_keys:
        precalc_emb_dict[meta_task_key] = torch.stack([item[meta_task_key] for item in precalculated_embeddings])
    precalc_emb = torch.mean(torch.stack(list(precalc_emb_dict.values())), dim=0)
    
    # Convert other data to tensors
    user_id = list(map(int, user_id))  # Convert user_id to integers first
    user_id = torch.tensor(user_id)
    # Convert to integer tensors, need padding
    # Get max length
    max_len = max(len(h) for h in history)
    # Create attention mask
    attention_mask = torch.zeros((len(history), max_len))
    padded_history = []
    
    # Pad sequences and create masks
    for i, h in enumerate(history):
        # Current sequence length
        curr_len = len(h)
        # Pad sequence
        # Left pad with 0s, right side is actual sequence
        padded_h = [0] * (max_len - curr_len) + h
        padded_history.append(padded_h)
        # Set mask, 0 for padding positions on left, 1 for actual values on right
        attention_mask[i, -curr_len:] = 1  # Use negative indexing to set mask from end
    history = torch.tensor(padded_history)
    history_mask = attention_mask
    target_item = torch.tensor(target_item)
    neg_items = torch.tensor(neg_items)
    pos_neg_items = torch.cat((target_item.unsqueeze(1), neg_items), dim=1)
    labels = torch.zeros(pos_neg_items.shape[0], dtype=torch.long)
    
    return user_id, history, history_mask, target_item, pos_neg_items, precalc_emb, labels

@hydra.main(version_base=None, config_path="config", config_name="em_train")
def main(cfg):
    """Main function for testing dataset and dataloader"""
    OmegaConf.resolve(cfg)

    missing_keys: Set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    set_seed(cfg.random_seed)
    train_dataset = MStepDataset(
        data_path=cfg.datasets.train_data_path,
        precalculated_embeddings_path=cfg.datasets.precalculated_embeddings_path,
        raw_data_path=cfg.datasets.raw_data_path,
        meta_task_keys=cfg.datasets.meta_task_keys,
        filter_keys=cfg.datasets.filter_keys,
        neg_sample_size=cfg.neg_sample_size,
        data_type="train"
    )
    print(f"Training set size: {len(train_dataset)}")
    
    # Test single sample
    user_id, history, target_item, neg_items, precalc_emb = train_dataset[0]
    print(f"\nSingle sample info:")
    print(f"User ID: {user_id}")
    print(f"History length: {len(history)}")
    print(f"Target item: {target_item}")
    print(f"Number of negative samples: {len(neg_items)}")
    print(f"Precalculated embedding keys: {precalc_emb.keys()}")
    print(f"Precalculated embedding dimension: {precalc_emb[cfg.datasets.meta_task_keys[0]].shape}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collect_fn_for_msstep
    )
    
    # Check one batch of data
    batch = next(iter(train_loader))
    user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
    print(f"\nBatch info:")
    print(f"User ID tensor shape: {user_ids.shape}")
    print(f"History tensor shape: {histories.shape}")
    print(f"History mask tensor shape: {history_mask.shape}")
    print(f"Target item tensor shape: {target_item.shape}")
    print(f"Positive/negative samples tensor shape: {pos_neg_items.shape}")
    for task_key, embeddings in precalculated_embeddings.items():
        print(f"{task_key} precalculated embedding tensor shape: {embeddings.shape}")
    print(f"Label tensor shape: {labels.shape}")
    
    # Create validation set
    val_dataset = MStepDataset(
        data_path=cfg.datasets.valid_data_path,
        precalculated_embeddings_path=cfg.datasets.precalculated_embeddings_path,
        raw_data_path=cfg.datasets.raw_data_path,
        meta_task_keys=cfg.datasets.meta_task_keys,
        filter_keys=cfg.datasets.filter_keys,
        neg_sample_size=cfg.neg_sample_size,
        data_type="val"
    )
    print(f"Validation set size: {len(val_dataset)}")
    
    # Test single validation sample
    user_id, history, target_item, neg_items, precalc_emb = val_dataset[0]
    print(f"\nValidation set single sample info:")
    print(f"User ID: {user_id}")
    print(f"History length: {len(history)}")
    print(f"Target item: {target_item}")
    print(f"Number of negative samples: {len(neg_items)}")
    print(f"Precalculated embedding dimension: {precalc_emb}")
    
    # Test validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collect_fn_for_msstep
    )
    
    # Check validation batch data
    val_batch = next(iter(val_loader))
    val_user_ids, val_histories, val_history_mask, val_target_item, val_pos_neg_items, val_precalculated_embeddings, val_labels = val_batch
    print(f"\nValidation batch info:")
    print(f"User ID tensor shape: {val_user_ids.shape}")
    print(f"History tensor shape: {val_histories.shape}")
    print(f"History mask tensor shape: {val_history_mask.shape}")
    print(f"Target item tensor shape: {val_target_item.shape}")
    print(f"Positive/negative samples tensor shape: {val_pos_neg_items.shape}")
    for task_key, embeddings in val_precalculated_embeddings.items():
        print(f"{task_key} precalculated embedding tensor shape: {embeddings.shape}")
    print(f"Label tensor shape: {val_labels.shape}")  


if __name__ == "__main__":
    main()