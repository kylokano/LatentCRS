import datetime
import getpass
import os
from typing import List
import torch.distributed as dist
import random
import numpy as np
import torch
import math

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# 创建一个简单的配置类来保存属性
class PrecalculatedRecModelConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # 确保所有必需的属性都存在
        required_attrs = [
            'item_size', 'hidden_size', 'max_seq_length',
            'hidden_dropout_prob', 'attention_probs_dropout_prob',
            'num_attention_heads', 'num_hidden_layers',
            'hidden_act', 'initializer_range'
        ]
        
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required configuration: {attr}")


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

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


class EarlyStopping:
    def __init__(self, patience=7, mode='max', verbose=False, delta=0, path='output/', trace_func=print, no_estep=False, no_mstep=False, max_saved=2):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path + ".pt"
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_path = None
        self.mode = mode
        self.no_estep = no_estep
        self.no_mstep = no_mstep
        self.max_saved = max_saved
        self.saved_models = []
        

    def __call__(self, val_loss, inference_model, prior_model):
        if self.mode == 'max': # 越大越好
            score = val_loss
        elif self.mode == 'min': # 越小越好
            score = -val_loss
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'max' or 'min'.")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, inference_model, prior_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping counter: {self.counter} which is greater than {self.patience}, early stop')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, inference_model, prior_model)
            self.counter = 0
            
    def refresh(self):
        # self.best_score = None
        self.counter = 0
        self.early_stop = False

    def save_checkpoint(self, val_loss, inference_model, prior_model):
        """保存验证集上表现最好的模型"""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        saved_path = self.path.replace('.pt', f'_{val_loss:.6f}.pt')
        self.best_model_path = saved_path
        self.trace_func(f"Saving model to {saved_path}")
        
        # 创建保存目录
        if not os.path.exists(os.path.dirname(saved_path)):
            os.makedirs(os.path.dirname(saved_path), exist_ok=True)
            
        # 保存当前模型
        if not self.no_estep:
            torch.save(inference_model.state_dict(), saved_path.replace('.pt', '_inference.pt'))
            torch.save(inference_model.state_dict(), self.path.replace('.pt', '_inference.pt'))
        if not self.no_mstep:
            torch.save(prior_model.state_dict(), saved_path.replace('.pt', '_prior.pt'))
            torch.save(prior_model.state_dict(), self.path.replace('.pt', '_prior.pt'))
            
        # 更新已保存模型列表
        self.saved_models.append(saved_path)
        
        # 如果超过最大保存数量，删除最旧的模型
        if len(self.saved_models) > self.max_saved:
            oldest_model = self.saved_models.pop(0)
            if os.path.exists(oldest_model.replace('.pt', '_inference.pt')):
                os.remove(oldest_model.replace('.pt', '_inference.pt'))
            if os.path.exists(oldest_model.replace('.pt', '_prior.pt')):
                os.remove(oldest_model.replace('.pt', '_prior.pt'))
                
        self.val_loss_min = val_loss

