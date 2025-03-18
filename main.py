import json
import os
from typing import Set
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import copy

from utils import rank0_print, set_seed, get_local_run_dir, PrecalculatedRecModelConfig, EarlyStopping
# from pre_rec_model.utils import load_clusters
from pre_rec_model.rec_models_cp import SASRecModel
from mydatasets import MStepDataset, collect_fn_for_msstep, get_useritem_rawid2modelid
from models import PosteriorNetwork, PriorNetwork
from trainers import EMTrainer
import numpy as np

# 定义 get_local_run_dir 函数
OmegaConf.register_new_resolver(
    "get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="em_train")
def main(cfg):
    OmegaConf.resolve(cfg)
    
    # 保存配置文件
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    config_path = os.path.join(cfg.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    set_seed(cfg.random_seed)
    
    if cfg.debug:
        cfg.epochs = 2
        cfg.batch_size = 2
        print(f"Debug mode, set epochs to {cfg.epochs}")
    
    
    # 加载预计算的Intention Embedding(KMeans)
    print("Loading Intention Embedding(KMeans)...")
    centers = np.load(os.path.join(cfg.datasets.precalculated_recmodel.kmeans_file_name, "centers.npy"))
    centers = torch.tensor(centers)
    print(f"Precalculated KMeans centers shape: {centers.shape}")
    
    cfg.intent_hidden = centers.shape[1]
    
    # 加载预计算的推荐模型
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
    
    missing_keys: Set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_logger = wandb.init(project=cfg.wandb.project, name=cfg.exp_name, config=wandb_config)
    wandb.define_metric("epoch")
    wandb.define_metric("train_epoch/*", step_metric="epoch")
    wandb.define_metric("valid_epoch")
    wandb.define_metric("valid/*", step_metric="valid_epoch")
    wandb.define_metric("m_steps")
    wandb.define_metric("train_epoch/m_step/*", step_metric="m_steps")
    wandb.define_metric("basic_model/*")
    
    # 创建训练集
    print("Creating training/validation dataset...")
    train_dataset = MStepDataset(
        data_path=cfg.datasets.train_data_path,
        precalculated_embeddings_path=cfg.datasets.precalculated_embeddings_path,
        raw_data_path=cfg.datasets.raw_data_path,
        meta_task_keys=cfg.datasets.meta_task_keys,
        filter_keys=cfg.datasets.filter_keys,
        neg_sample_size=cfg.neg_sample_size,
        id2modelid=id2modelid,
        data_type="train",
        is_debug=cfg.debug
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size       ,  # 您可能想要从配置文件中读取batch_siz额
        collate_fn=collect_fn_for_msstep,
        num_workers=4,   # 您可能想要从配置文件中读取num_workers
        sampler=RandomSampler(train_dataset)
    )
    
    # 创建验证集
    val_dataset = MStepDataset(
        data_path=cfg.datasets.valid_data_path,
        precalculated_embeddings_path=cfg.datasets.precalculated_embeddings_path,
        raw_data_path=cfg.datasets.raw_data_path,
        meta_task_keys=cfg.datasets.meta_task_keys,
        filter_keys=cfg.datasets.filter_keys,
        neg_sample_size=cfg.neg_sample_size,
        id2modelid=id2modelid,
        data_type="valid",
        is_debug=cfg.debug
    )
    print("Validation dataset created.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,  # 您可能想要从配置文件中读取batch_size
        collate_fn=collect_fn_for_msstep,
        num_workers=4,   # 您可能想要从配置文件中读取num_workers
        sampler=SequentialSampler(val_dataset)
    )
    print("Data loaders created.")
    
    
    test_dataset = MStepDataset(
        data_path=cfg.datasets.test_data_path,
        precalculated_embeddings_path=cfg.datasets.precalculated_embeddings_path,
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
        sampler=SequentialSampler(test_dataset)
    )
    
    prior_model = PriorNetwork(cfg, centers, precalc_recmodel)
    if cfg.same_prerec_model:
        posterior_model = PosteriorNetwork(cfg, centers, copy.deepcopy(precalc_recmodel))
    else:
        posterior_model = PosteriorNetwork(cfg, centers, precalc_recmodel)
    early_stop = EarlyStopping(cfg.early_stop.patience, cfg.early_stop.mode, cfg.early_stop.verbose, cfg.early_stop.delta, cfg.early_stop.path+"/"+cfg.exp_name, no_estep=cfg.no_estep, no_mstep=cfg.no_mstep)
    
    if cfg.posterior.load_pretrained:
        print(f"Loading pretrained inference model from {cfg.posterior.pretrained_path}")
        pretrained_model = torch.load(cfg.posterior.pretrained_path, weights_only=True)
        posterior_model.load_state_dict(pretrained_model)
        print("Pretrained model loaded.")
    
    if cfg.prior.load_pretrained:
        print(f"Loading pretrained prior model from {cfg.prior.pretrained_path}")
        pretrained_model = torch.load(cfg.prior.pretrained_path, weights_only=True)
        prior_model.load_state_dict(pretrained_model)
        print("Pretrained model loaded.")
    
    print(f"cfg.same_y_m: {cfg.same_y_m}")
    if cfg.same_y_m:
        print("Loading pretrained inference model y_m to prior model...")
        prior_model.linear_intent.load_state_dict(posterior_model.linear_value.state_dict())
    
    raw_inited_params, precaled_prior_params, precaled_posterior_params = [], [], []
    optim_parameters_list = []
    for name, param in prior_model.named_parameters():
        if "user_sequence_model" in name:
            precaled_prior_params.append(param)
        else:
            raw_inited_params.append(param)
    for name, param in posterior_model.named_parameters():
        if "user_sequence_model" in name:
            precaled_posterior_params.append(param)
        else:
            raw_inited_params.append(param)
    if cfg.prior.use_pretrained_intent:
        optim_parameters_list.append({"params": precaled_prior_params, "lr": min(cfg.learning_rate/cfg.pretrained_lr_ratio, 0.0001)})
    else:
        optim_parameters_list.append({"params": precaled_prior_params, "lr": cfg.learning_rate})

    if cfg.posterior.use_pretrained_intent:
        if not cfg.same_prerec_model:
            optim_parameters_list.append({"params": precaled_posterior_params, "lr": min(cfg.learning_rate/cfg.pretrained_lr_ratio, 0.0001)})
    else:
        optim_parameters_list.append({"params": precaled_posterior_params, "lr": cfg.learning_rate})
    optim_parameters_list.append({"params": raw_inited_params, "lr": cfg.learning_rate})
        
    optimizer = torch.optim.Adam(optim_parameters_list)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=cfg.scheduler.start_factor, end_factor=cfg.scheduler.end_factor, total_iters=cfg.scheduler.total_iters)
    trainer = EMTrainer(cfg, prior_model, posterior_model, optimizer, scheduler, train_loader, val_loader, test_loader, early_stop, wandb_logger, cfg)
    
    # trainer.test_basic_prerec_model(precalc_recmodel)
    
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()