from utils import recall_at_k, ndcg_k
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

class BasicTrainer:
    def __init__(self, config, prior_model, inference_model, optimizer, scheduler, train_loader, val_loader, test_loader, early_stop, cfg):
        pass

    def get_full_sort_score(self, epoch, prefix, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "HIT@5": recall[0],
            "NDCG@5": ndcg[0],
            "HIT@10": recall[1],
            "NDCG@10": ndcg[1],
            "HIT@15": recall[2],
            "NDCG@15": ndcg[2],
            "HIT@20": recall[3],
            "NDCG@20": ndcg[3],
        }
        print(epoch, prefix, post_fix)
        return post_fix


class EMTrainer(BasicTrainer):
    def __init__(self, config, prior_model, inference_model, optimizer, scheduler, train_loader, val_loader, test_loader, early_stop, wandb_logger, cfg):
        super().__init__(config, prior_model, inference_model, optimizer, scheduler, train_loader, val_loader, test_loader, early_stop, cfg)
        self.prior_model = prior_model
        self.inference_model = inference_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.early_stop = early_stop
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.num_items = self.prior_model.item_embedding.weight.shape[0]
        self.wandb_logger = wandb_logger
        self.neg_loss = nn.CrossEntropyLoss(reduction='none')
        

        if self.cuda_condition:
            self.prior_model.to(self.device)
            self.inference_model.to(self.device)

        betas = (self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2)
        self.optim = Adam([{"params":self.prior_model.parameters(), "name":"prior"}, {"params":self.inference_model.parameters(), "name":"inference"}], lr=self.cfg.learning_rate, betas=betas, weight_decay=self.cfg.optim.adam_weight_decay)

        prior_params = sum([p.nelement() * p.element_size() for p in self.prior_model.parameters()]) / (1024 * 1024)
        inference_params = sum([p.nelement() * p.element_size() for p in self.inference_model.parameters()]) / (1024 * 1024)
        print(f"Prior Model Parameters: {prior_params:.2f}MB")
        print(f"Inference Model Parameters: {inference_params:.2f}MB")
        
        
    def calculate_elbo(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        q = self.inference_model.distribution_m_u(histories, history_mask) # [batch_size, num_intent]
        
        target_item = target_item.unsqueeze(1) # [batch_size, 1]
        pm_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        logq, logp_m_ux = torch.log(q + 1e-5), torch.log(pm_ux + 1e-5)

        elbo = (q * (logp_m_ux - logq)).sum(dim=-1) # [batch_size]
        return elbo
    
    def calculate_elbo_with_y(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        target_item = target_item.unsqueeze(1) # [batch_size, 1]
        p_y_m = self.prior_model.get_y_m(target_item) # [batch_size, num_intent, 1]
        m_yux = p_y_m.squeeze() * p_m_ux # [batch_size, num_intent]
        p_m_yux = torch.softmax(m_yux, dim=-1) # [batch_size, num_intent]
        p = p_m_yux
        # p = self.prior_model.get_m_yux(histories, history_mask, precalculated_embeddings, target_item) # [batch_size, num_intent]
        
        q = self.inference_model.distribution_m_u(histories, history_mask) # [batch_size, num_intent]
        logq, logp = torch.log(q + 1e-5), torch.log(p + 1e-5)
        
        elbo = (q * (logp - logq)).sum(dim=-1) # [batch_size]
        return elbo
    
    def calculate_elbo_with_y_negative(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        q = self.inference_model.distribution_m_u(histories, history_mask) # [batch_size, num_intent]
        
        py_m = self.prior_model.get_y_m(pos_neg_items).squeeze() # [batch_size, num_intent, num_items]
        py_m = torch.softmax(py_m, dim=-1) # [batch_size, num_intent, num_items]

        labels_expanded = labels.view(labels.shape[0], 1).expand(-1, py_m.shape[1])  # [120, 128]
        labels_expanded = labels_expanded.unsqueeze(2)  # [120, 128, 1]
        extracted_values = torch.gather(py_m, 2, labels_expanded)  # [120, 128, 1]
        extracted_values = extracted_values.squeeze(2)  # [120, 128]
        py_m_target = extracted_values # [batch_size, num_intent]

        pm_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]

        
        scores = py_m_target * pm_ux
        log_m_yux = torch.log_softmax(scores, dim=-1)
        
        logq = torch.log(q + 1e-5)
        elbo = (q * (log_m_yux - logq)).sum(dim=-1)
        return elbo
    
    def calculate_elbo_with_negetive(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        return self.elbo(histories, history_mask, pos_neg_items, precalculated_embeddings), labels
    
    def calculate_logp_with_negetive(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = self.prior_model.get_y_m(pos_neg_items, intent_distribution=p_m_ux) # [batch_size, num_items, num_intent]
        
        return p_y, labels
    
    def calculate_q_scores(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        q_m_u, reconstruct_loss = self.inference_model.get_predict(histories, history_mask, pos_neg_items)
        return q_m_u, reconstruct_loss, labels
    
    def calculate_m_infonce(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = self.prior_model.get_y_m(pos_neg_items) # [batch_size, num_intent, num_items]
        # p_y = torch.softmax(p_y, dim=1) # [batch_size, num_intent, num_items]
        y_mux = p_y * p_m_ux.unsqueeze(-1) # [batch_size, num_intent, num_items]
        # y_mux = self.prior_model.get_y_mux(histories, history_mask, precalculated_embeddings, pos_neg_items) # batch_size, num_intent, num_items
        y_mux = y_mux.transpose(1, 2) # batch_size, num_items, num_intent
        
        
        
        
        labels = labels.unsqueeze(1).expand(-1, y_mux.shape[2]) # batch_size, num_intent
        infonce_loss = self.neg_loss(y_mux, labels) # batch_size, num_intent
        q_m_u = self.inference_model.distribution_m_u(histories, history_mask) # batch_size, num_intent
        infonce_loss = q_m_u * infonce_loss # batch_size, num_intent
        infonce_loss = infonce_loss.mean(dim=-1).mean(dim=-1) # 1
        
        # infonce_loss = infonce_loss.mean(dim=-1).mean(dim=-1) # 1
        return infonce_loss
    
    def calculate_m_rec_loss(self, batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = batch
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = self.prior_model.get_y_m(pos_neg_items, intent_distribution=p_m_ux) # [batch_size, num_items]
        rec_loss = self.neg_loss(p_y, labels)# [batch_size, num_items]
        rec_loss = rec_loss.mean(dim=-1).mean(dim=-1) # 1
        return rec_loss 
    
    
    def train_elbo_same_kl(self, epoch, train_dataloader):
        mstep_dataloader = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="M-step")
        estep_dataloader = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="E-step")
        total_m_step_elbo, total_m_step_loss, total_m_step_neg_loss = 0, 0, 0
        total_e_step_elbo, total_e_step_loss = 0, 0
        epoch_metrics = {}
        
        # E-step
        if not self.cfg.no_estep:
            self.prior_model.eval()
            self.inference_model.train()
            for param in self.prior_model.parameters():
                param.requires_grad = True
            for param in self.inference_model.parameters():
                param.requires_grad = True
            if not self.cfg.train_preintent:
                for param in self.inference_model.user_sequence_model.parameters():
                    param.requires_grad = False
            for i, batch in estep_dataloader:
                batch = [value.to(self.device) for value in batch]
                elbo = self.calculate_elbo_with_y(batch)
                total_e_step_elbo += elbo.mean().squeeze().item()
                if self.cfg.use_negetive_loss and self.cfg.e_negetive_loss_weight > 0:
                    scores, reconstruct_loss, labels = self.calculate_q_scores(batch)
                    neg_loss = self.neg_loss(scores/self.cfg.neg_loss_temp, labels).mean()
                    if epoch >= self.cfg.only_rec_epoch:
                        loss = -elbo.mean() + self.cfg.e_negetive_loss_weight * neg_loss + self.cfg.lambda_reconstruct * reconstruct_loss.mean()
                    else:
                        loss = self.cfg.e_negetive_loss_weight * neg_loss + self.cfg.lambda_reconstruct * reconstruct_loss.mean()
                else:
                    loss = -elbo.mean()
                
                if self.cfg.m_rec_loss_weight <= 0 and epoch < self.cfg.only_rec_epoch:
                    raise ValueError("m_rec_loss_weight must be greater than 0 when only_rec_epoch is not reached")
                
                if epoch >= self.cfg.only_rec_epoch:
                    elbo = self.calculate_elbo(batch)
                    loss += -elbo.mean()
                    total_m_step_elbo += elbo.mean().squeeze().item()
                    if self.cfg.m_negetive_loss_weight > 0:
                        neg_loss = self.calculate_m_infonce(batch)
                        loss += self.cfg.m_negetive_loss_weight * neg_loss
                        total_m_step_elbo += elbo.mean().squeeze().item() + neg_loss.item()
                        
                if self.cfg.m_rec_loss_weight > 0:
                    rec_loss = self.calculate_m_rec_loss(batch)
                    if epoch >= self.cfg.only_rec_epoch:
                        loss += self.cfg.m_rec_loss_weight * rec_loss
                    else:
                        loss = self.cfg.m_rec_loss_weight *rec_loss
                    total_m_step_neg_loss += rec_loss.item()
                
                self.optim.zero_grad()
                total_e_step_loss += loss.item()
                if self.cfg.wandb.log_steps:
                    self.wandb_logger.log({"train/e_step/elbo_per_step": elbo.mean().squeeze().item(), "train/e_step/loss_per_step": loss.item(), "e_steps": epoch*len(train_dataloader)+i})
                loss.backward()
                self.optim.step()
            epoch_metrics["train_epoch/estep_elbo"] = total_e_step_elbo / len(train_dataloader)
            epoch_metrics["train_epoch/estep_loss"] = total_e_step_loss / len(train_dataloader)
            print("Epoch", epoch, "E-step ELBO:", total_e_step_elbo / len(train_dataloader), "E-step Loss:", total_e_step_loss / len(train_dataloader))
            epoch_metrics = {"train_epoch/mstep_elbo": total_m_step_elbo / len(train_dataloader), "train_epoch/mstep_loss": total_m_step_loss / len(train_dataloader), "train_epoch/mstep_neg_loss": total_m_step_neg_loss / len(train_dataloader), "epoch": epoch}
            print("Epoch", epoch, "M-step ELBO:", total_m_step_elbo / len(train_dataloader), "M-step Loss:", total_m_step_loss / len(train_dataloader))
        
        
        self.wandb_logger.log(epoch_metrics)
        self.scheduler.step()
    
    def train_elbo_only_positive_iteration(self, epoch, train_dataloader):
        mstep_dataloader = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="M-step")
        estep_dataloader = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="E-step")
        total_m_step_elbo, total_m_step_loss, total_m_step_neg_loss = 0, 0, 0
        total_e_step_elbo, total_e_step_loss = 0, 0
        epoch_metrics = {}
        
        # E-step
        if not self.cfg.no_estep:
            self.prior_model.eval()
            self.inference_model.train()
            for param in self.prior_model.parameters():
                param.requires_grad = False
            for param in self.inference_model.parameters():
                param.requires_grad = True
            if not self.cfg.train_preintent:
                for param in self.inference_model.user_sequence_model.parameters():
                    param.requires_grad = False
            for i, batch in estep_dataloader:
                batch = [value.to(self.device) for value in batch]
                elbo = self.calculate_elbo_with_y(batch)
                total_e_step_elbo += elbo.mean().squeeze().item()
                if self.cfg.use_negetive_loss and self.cfg.e_negetive_loss_weight > 0:
                    scores, reconstruct_loss, labels = self.calculate_q_scores(batch)
                    neg_loss = self.neg_loss(scores/self.cfg.neg_loss_temp, labels).mean()
                    if epoch >= self.cfg.only_rec_epoch:
                        loss = -elbo.mean() + self.cfg.e_negetive_loss_weight * neg_loss + self.cfg.lambda_reconstruct * reconstruct_loss.mean()
                    else:
                        loss = self.cfg.e_negetive_loss_weight * neg_loss + self.cfg.lambda_reconstruct * reconstruct_loss.mean()
                else:
                    loss = -elbo.mean()
                self.optim.zero_grad()
                total_e_step_loss += loss.item()
                if self.cfg.wandb.log_steps:
                    self.wandb_logger.log({"train/e_step/elbo_per_step": elbo.mean().squeeze().item(), "train/e_step/loss_per_step": loss.item(), "e_steps": epoch*len(train_dataloader)+i})
                loss.backward()
                self.optim.step()
            epoch_metrics["train_epoch/estep_elbo"] = total_e_step_elbo / len(train_dataloader)
            epoch_metrics["train_epoch/estep_loss"] = total_e_step_loss / len(train_dataloader)
            print("Epoch", epoch, "E-step ELBO:", total_e_step_elbo / len(train_dataloader), "E-step Loss:", total_e_step_loss / len(train_dataloader))
        
        # M-step
        if not self.cfg.no_mstep:
            self.prior_model.train()
            self.inference_model.eval()
            for param in self.prior_model.parameters():
                param.requires_grad = True
            for param in self.inference_model.parameters():
                param.requires_grad = False
            if self.cfg.same_y_m:
                for param in self.prior_model.linear_intent.parameters():
                    param.requires_grad = False
            if not self.cfg.train_preintent:
                for param in self.prior_model.user_sequence_model.parameters():
                    param.requires_grad = False
            elbo = np.zeros(shape=(2,1))
            neg_loss = np.zeros(1)
            for i, batch in mstep_dataloader:
                batch = [value.to(self.device) for value in batch]
                if self.cfg.m_rec_loss_weight <= 0 and epoch < self.cfg.only_rec_epoch:
                    raise ValueError("m_rec_loss_weight must be greater than 0 when only_rec_epoch is not reached")
                
                if epoch >= self.cfg.only_rec_epoch:
                    elbo = self.calculate_elbo(batch)
                    loss = -elbo.mean()
                    total_m_step_elbo += elbo.mean().squeeze().item()
                    if self.cfg.m_negetive_loss_weight > 0:
                        neg_loss = self.calculate_m_infonce(batch)
                        loss += self.cfg.m_negetive_loss_weight * neg_loss
                        total_m_step_elbo += elbo.mean().squeeze().item() + neg_loss.item()
                        
                if self.cfg.m_rec_loss_weight > 0:
                    rec_loss = self.calculate_m_rec_loss(batch)
                    if epoch >= self.cfg.only_rec_epoch:
                        loss += self.cfg.m_rec_loss_weight * rec_loss
                    else:
                        loss = self.cfg.m_rec_loss_weight *rec_loss
                    total_m_step_neg_loss += rec_loss.item()
                total_m_step_loss += loss.item()
                
                if self.cfg.wandb.log_steps:
                    self.wandb_logger.log({"train/m_step/elbo_per_step": elbo.mean().squeeze().item(), "train/m_step/loss_per_step": loss.item(), "train/m_step/neg_loss_per_step": neg_loss.item(), "m_steps": epoch*len(train_dataloader)+i})
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            epoch_metrics = {"train_epoch/mstep_elbo": total_m_step_elbo / len(train_dataloader), "train_epoch/mstep_loss": total_m_step_loss / len(train_dataloader), "train_epoch/mstep_neg_loss": total_m_step_neg_loss / len(train_dataloader), "epoch": epoch}
            print("Epoch", epoch, "M-step ELBO:", total_m_step_elbo / len(train_dataloader), "M-step Loss:", total_m_step_loss / len(train_dataloader))
        
        
        self.wandb_logger.log(epoch_metrics)
        self.scheduler.step()
        
    def item_embedding_without_text(self, histories, history_mask):
        intent_distribution = self.inference_model.distribution_m_u(histories, history_mask) # [batch_size, num_intent]
        pred_item_embedding = self.prior_model.distribution_y_um(histories, history_mask, intent_distribution, "InferenceStep") # [batch_size, item_embedding_dim]
        if self.cfg.prior.y_um.e_step_distribution == "gaussian":
            pred_item_embedding, pred_item_logvar = pred_item_embedding.chunk(2, dim=-1)
        return pred_item_embedding
    
    def predict_embedding_similarity(self, pred_item_embedding, all_item_embedding):
        similarity = torch.matmul(pred_item_embedding, all_item_embedding.transpose(0, 1))
        return similarity
    
    def get_top20_item(self, scores):
        ind_without_text = np.argpartition(scores, -20)[:, -20:]
        arr_ind_without_text = scores[np.arange(len(scores))[:, None], ind_without_text]
        arr_ind_argsort_without_text = np.argsort(arr_ind_without_text)[:, ::-1]
        batch_pred_list_without_text = ind_without_text[np.arange(len(scores))[:, None], arr_ind_argsort_without_text]
        return batch_pred_list_without_text
    
    def full_sort_predict_embedding(self, batch_idx, test_batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        # without text
        pred_item_embedding_without_text = self.item_embedding_without_text(histories, history_mask)
        # with text
        pred_item_embedding_with_text = self.prior_model.item_embedding_predict_step(histories, history_mask, precalculated_embeddings)
        
        all_item_embedding = self.prior_model.item_embedding.weight
        pred_score_without_text = self.predict_embedding_similarity(pred_item_embedding_without_text, all_item_embedding)
        pred_score_with_text = self.predict_embedding_similarity(pred_item_embedding_with_text, all_item_embedding)
        
        truely_item = target_item.cpu().data.numpy().copy()
        truely_item = [[i] for i in truely_item]
        pred_score_without_text = pred_score_without_text.cpu().data.numpy().copy()
        pred_score_with_text = pred_score_with_text.cpu().data.numpy().copy()
        
        batch_pred_list_without_text = self.get_top20_item(pred_score_without_text)
        batch_pred_list_with_text = self.get_top20_item(pred_score_with_text)
        return batch_pred_list_without_text, batch_pred_list_with_text, truely_item
    
    def pridict_with_weighted(self, user_ids, history, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels):
        p_m_ux = self.prior_model.distribution_m_ux(history, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        target_item = torch.arange(self.num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
        p_y = self.prior_model.get_y_m(target_item) # [batch_size, num_intent, num_items]
        y_mux = p_y * p_m_ux.unsqueeze(-1) # [batch_size, num_intent, num_items]
        m_ux = p_m_ux.unsqueeze(-1) # batch_size, num_intent, 1
        y_mux = y_mux * m_ux # batch_size, num_intent, num_items
        p_y_mux = y_mux.sum(dim=1) # batch_size, num_items
        return p_y_mux
        
    
    def full_sort_predict_elbo(self, batch_idx, test_batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        truely_item = target_item.cpu().data.numpy().copy()
        truely_item = [[i] for i in truely_item]
        
        target_item = torch.arange(self.num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = self.prior_model.get_y_m(target_item, intent_distribution=p_m_ux) # [batch_size, num_items]
        scores = p_y
        
        scores_without_text, reconstruct_loss = self.inference_model.get_predict(histories, history_mask, target_item)
        user_seq_predict = self.prior_model.get_user_seq_predict(histories, history_mask, target_item)
        
        scores = scores.cpu().data.numpy().copy()
        scores_without_text = scores_without_text.cpu().data.numpy().copy()
        user_seq_predict_score = user_seq_predict.cpu().data.numpy().copy()
        
        top20_item = self.get_top20_item(scores)
        top20_item_without_text = self.get_top20_item(scores_without_text)
        top20_item_user_seq_predict = self.get_top20_item(user_seq_predict_score)
        return top20_item, top20_item_without_text, truely_item, top20_item_user_seq_predict
    
    def full_sort_predict_elbo_small(self, batch_idx, test_batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        truely_item = target_item.cpu().data.numpy().copy()
        truely_item = [[i] for i in truely_item]
        
        target_item = torch.arange(self.num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
        p_m_ux = self.prior_model.distribution_m_ux(histories, history_mask, precalculated_embeddings) # [batch_size, num_intent]
        p_y = self.prior_model.get_y_m(target_item, intent_distribution=p_m_ux) # [batch_size, num_items]
        scores = p_y
        scores = scores.cpu().data.numpy().copy()
        top20_item = self.get_top20_item(scores)
        return top20_item, truely_item
    
    def full_sort_without_weight(self, batch_idx, test_batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        
        
        y_mux = self.pridict_with_weighted(user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels)
        
        y_mux = y_mux.cpu().data.numpy().copy()
        top20_item = self.get_top20_item(y_mux)
        return top20_item
    
    def full_sort_y_qm(self, batch_idx, test_batch):
        user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
        target_item = torch.arange(self.num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
        attention_weights = self.inference_model.distribution_m_u(histories, history_mask)
        scores = self.prior_model.get_y_m(target_item, intent_distribution=attention_weights)
        
        scores = scores.cpu().data.numpy().copy()
        
        top20_item = self.get_top20_item(scores)
        return top20_item
    
    def full_sort_predict(self, epoch_idx, test_loader):
        self.prior_model.eval()
        self.inference_model.eval()
        for batch_idx, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Full-sort predict"):
            test_batch = [value.to(self.device) for value in test_batch]
            top20_item, top20_item_without_text, truely_item, user_seq_predict = self.full_sort_predict_elbo(batch_idx, test_batch)
            
            if batch_idx == 0:
                logits_pred_list = top20_item
                logits_pred_list_without_text = top20_item_without_text
                answer_list = truely_item
                user_seq_predict_list = user_seq_predict
            else:
                logits_pred_list = np.append(logits_pred_list, top20_item, axis=0)
                logits_pred_list_without_text = np.append(logits_pred_list_without_text, top20_item_without_text, axis=0)
                answer_list = np.append(answer_list, truely_item, axis=0)
                user_seq_predict_list = np.append(user_seq_predict_list, user_seq_predict, axis=0)
                
        logits_pred = self.get_full_sort_score(epoch_idx, "Logits", answer_list, logits_pred_list)
        logits_pred_without_text = self.get_full_sort_score(epoch_idx, "Logits without text", answer_list, logits_pred_list_without_text)
        user_seq_predict = self.get_full_sort_score(epoch_idx, "User sequence predict", answer_list, user_seq_predict_list)
        
        if type(epoch_idx) == int:
            metrics_dict = {}
            for key, value in logits_pred.items():
                metrics_dict[f"valid/Logits/{key}"] = value
            for key, value in logits_pred_without_text.items():
                metrics_dict[f"valid/Logits without text/{key}"] = value
            for key, value in user_seq_predict.items():
                metrics_dict[f"valid/User sequence predict/{key}"] = value
            metrics_dict[f"valid_epoch"] = epoch_idx
            self.wandb_logger.log(metrics_dict)
        else:
            assert epoch_idx == "test"
            metrics_dict = {}
            for key, value in logits_pred.items():
                metrics_dict[f"eval/Logits/{key}"] = value
            for key, value in logits_pred_without_text.items():
                metrics_dict[f"eval/Logits without text/{key}"] = value
            for key, value in user_seq_predict.items():
                metrics_dict[f"eval/User sequence predict/{key}"] = value
            self.wandb_logger.log(metrics_dict)
        return logits_pred_without_text, logits_pred
    
    def full_sort_predict_small(self, epoch_idx, test_loader):
        self.prior_model.eval()
        self.inference_model.eval()
        print(len(test_loader))
        for batch_idx, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Full-sort predict"):
            test_batch = [value.to(self.device) for value in test_batch]
            top20_item, truely_item = self.full_sort_predict_elbo_small(batch_idx, test_batch)
            if batch_idx == 0:
                logits_pred_list = top20_item
                answer_list = truely_item
            else:
                logits_pred_list = np.append(logits_pred_list, top20_item, axis=0)
                answer_list = np.append(answer_list, truely_item, axis=0)
        
        logits_pred = self.get_full_sort_score(epoch_idx, "Logits", answer_list, logits_pred_list)
        if type(epoch_idx) == int:
            metrics_dict = {}
            for key, value in logits_pred.items():
                metrics_dict[f"valid/Logits/{key}"] = value
            metrics_dict[f"valid_epoch"] = epoch_idx
            self.wandb_logger.log(metrics_dict)
        else:
            assert epoch_idx == "test"
            metrics_dict = {}
            for key, value in logits_pred.items():
                metrics_dict[f"eval/Logits/{key}"] = value
            self.wandb_logger.log(metrics_dict)
        return {}, logits_pred
        
        
    def train(self):
        if self.cfg.validation_first:
            self.full_sort_predict(0, self.val_loader)
        for epoch in range(self.cfg.epochs):
            print(f"Epoch {epoch} of {self.cfg.epochs}")
            self.train_elbo_only_positive_iteration(epoch, self.train_loader)
            logits_pred_without_text, logits_pred = self.full_sort_predict(epoch, self.val_loader)
            if self.early_stop.early_stop:
                break
            stop_metrics = self.cfg.early_stop.stop_metrics
            if self.cfg.early_stop.stopbywhich == "EStep":
                self.early_stop(logits_pred_without_text[stop_metrics], self.inference_model, self.prior_model)
            elif self.cfg.early_stop.stopbywhich == "MStep":
                self.early_stop(logits_pred[stop_metrics], self.inference_model, self.prior_model)
            else:
                raise ValueError("Early stop by which is not supported, should be EStep or MStep")
            if self.early_stop.early_stop:
                break
        if self.cfg.load_best_model:
            best_model_path = self.early_stop.best_model_path
            if best_model_path is None:
                raise ValueError("Best model path is None, please check the early stopping settings")
            print(f"Loading best model from {best_model_path}")
            if not self.cfg.no_estep:
                self.inference_model.load_state_dict(torch.load(best_model_path.replace('.pt', '_inference.pt'), weights_only=True))
            if not self.cfg.no_mstep:
                self.prior_model.load_state_dict(torch.load(best_model_path.replace('.pt', '_prior.pt'), weights_only=True))
    
    
    def test(self):
        logits_pred_list_without_text, logits_pred = self.full_sort_predict("test", self.test_loader)
        return logits_pred_list_without_text, logits_pred
        
    def test_basic_prerec_model(self, pre_sas_model):
        test_dataloader = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Test PreRec Model")
        pre_sas_model.eval()
        for batch_idx, test_batch in test_dataloader:
            test_batch = [value.to(self.device) for value in test_batch]
            user_ids, histories, history_mask, target_item, pos_neg_items, precalculated_embeddings, labels = test_batch
            all_item = torch.arange(self.num_items).to("cuda").unsqueeze(0).expand(user_ids.shape[0], -1)
            user_embeddings = pre_sas_model(histories, history_mask)[:, -1, :]
            target_items_embedding = pre_sas_model.item_embeddings(all_item) # [batch_size, num_items, item_embedding_dim]
            user_embeddings = user_embeddings.unsqueeze(1) # [batch_size, 1, user_embedding_dim]
            scores = torch.matmul(user_embeddings, target_items_embedding.transpose(1, 2)) # [batch_size, 1, num_items]
            scores = scores.squeeze(1) # [batch_size, num_items]
            scores = scores.cpu().data.numpy().copy()
            top20_item = self.get_top20_item(scores)
            truely_item = target_item.cpu().data.numpy().copy()
            truely_item = [[i] for i in truely_item]
            
            if batch_idx == 0:
                top20_item_list = top20_item
                answer_list = truely_item
            else:
                top20_item_list = np.append(top20_item_list, top20_item, axis=0)
                answer_list = np.append(answer_list, truely_item, axis=0)
            
                
        scores = self.get_full_sort_score("test_basic", "bisic_model", answer_list, top20_item_list)
        metrics_dict = {}
        for key, value in scores.items():
            metrics_dict[f"basic_model/{key}"] = value
        self.wandb_logger.log(metrics_dict)
        print("test_basic_model", metrics_dict)
        return top20_item
