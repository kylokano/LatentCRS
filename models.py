import torch
import torch.nn as nn
import torch.nn.functional as F
from pre_rec_model.rec_models_cp import SASRecModel
import math
# Prior Network p(m|u,x) and p(y|u,m)
class PriorNetwork(nn.Module):
    def __init__(self, config, intent_embedding:torch.Tensor, sas_rec_model:SASRecModel):
        super(PriorNetwork, self).__init__()
        self.config = config
        if self.config.prior.use_pretrained_intent:
            print("Use Pretrained seqrec model for Prior Network")
            self.user_sequence_model = sas_rec_model
        else:
            self.user_sequence_model = SASRecModel(config.precalculated_recmodel)
        self.intent_embedding = nn.Parameter(intent_embedding)
        self.item_embedding = self.user_sequence_model.item_embeddings
        self.config = config
        
        self.m_ux_agg_type = config.prior.m_ux.agg_type
        self.m_ux_layer_hidden = config.prior.m_ux.layer_hidden + [config.intent_hidden]
        self.m_ux_num_layers = len(self.m_ux_layer_hidden)
        self.m_ux_layer_norm = config.prior.m_ux.layer_norm
        self.m_ux_hidden_dropout_prob = config.prior.m_ux.hidden_dropout_prob
        
        if config.prior.m_ux.use_text_projection:
            self.text_projection = config.prior.m_ux.text_projection
            self.m_ux_projection = nn.Sequential(
                nn.Linear(config.text_embedding_dim, self.text_projection[0], bias=False),
                nn.Dropout(config.prior.m_ux.hidden_dropout_prob),
                nn.LayerNorm(self.text_projection[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.text_projection[0], self.text_projection[1], bias=True),
            )
            m_ux_first_hidden = self.text_projection[1] + config.precalculated_recmodel.hidden_size
        else:
            m_ux_first_hidden = config.text_embedding_dim + config.precalculated_recmodel.hidden_size
        
        if self.m_ux_agg_type == "concat":
            self.m_ux_layer_hidden = [m_ux_first_hidden] + self.m_ux_layer_hidden
            self.m_ux_agg_layer = nn.ModuleList([nn.Linear(self.m_ux_layer_hidden[i], self.m_ux_layer_hidden[i+1]) for i in range(self.m_ux_num_layers)])
            self.m_ux_layer_norm = nn.ModuleList([nn.LayerNorm(self.m_ux_layer_hidden[i]) for i in range(self.m_ux_num_layers)])
        elif self.m_ux_agg_type == "gated":
            if config.prior.m_ux.use_text_projection:
                self.m_ux_linear_query = nn.Linear(self.text_projection[1], config.precalculated_recmodel.hidden_size)
                self.m_ux_linear_query_2 = nn.Linear(self.text_projection[1], config.precalculated_recmodel.hidden_size)
            else:
                self.m_ux_linear_query = nn.Linear(config.text_embedding_dim, config.precalculated_recmodel.hidden_size)
                self.m_ux_linear_query_2 = nn.Linear(config.text_embedding_dim, config.precalculated_recmodel.hidden_size)
            self.m_ux_linear_key = nn.Linear(config.intent_hidden, config.intent_hidden)
            self.m_ux_linear_key_2 = nn.Linear(config.intent_hidden, config.intent_hidden)
            self.m_ux_layer_hidden = [config.intent_hidden] + self.m_ux_layer_hidden
            self.m_ux_agg_layer = nn.ModuleList([nn.Linear(self.m_ux_layer_hidden[i], self.m_ux_layer_hidden[i+1]) for i in range(self.m_ux_num_layers)])
            self.m_ux_layer_norm = nn.ModuleList([nn.LayerNorm(self.m_ux_layer_hidden[i+1]) for i in range(self.m_ux_num_layers)])
            
        
        self.y_um_agg_type = config.prior.y_um.agg_type
        if self.config.prior.y_um.e_step_distribution == "gaussian":
            self.y_um_layer_hidden = config.prior.y_um.layer_hidden + [config.precalculated_recmodel.hidden_size * 2]
        else:
            self.y_um_layer_hidden = config.prior.y_um.layer_hidden + [config.precalculated_recmodel.hidden_size]
        self.y_um_num_layers = len(self.y_um_layer_hidden)
        self.y_um_layer_norm = config.prior.y_um.layer_norm
        self.y_um_hidden_dropout_prob = config.prior.y_um.hidden_dropout_prob
        
        if self.y_um_agg_type == "concat":
            self.y_um_layer_hidden = [config.intent_hidden + self.item_embedding.embedding_dim] + self.y_um_layer_hidden
            self.y_um_agg_layer = nn.ModuleList([nn.Linear(self.y_um_layer_hidden[i], self.y_um_layer_hidden[i+1]) for i in range(self.y_um_num_layers)])
            self.y_um_layer_norm = nn.ModuleList([nn.LayerNorm(self.y_um_layer_hidden[i]) for i in range(self.y_um_num_layers)])
        elif self.y_um_agg_type == "gated":
            self.projection_intent2item = nn.Linear(config.intent_hidden, self.item_embedding.embedding_dim)
            self.gate_layer = nn.Linear(self.item_embedding.embedding_dim * 2, self.item_embedding.embedding_dim)
            self.y_um_layer_hidden = [self.item_embedding.embedding_dim] + self.y_um_layer_hidden
            self.y_um_agg_layer = nn.ModuleList([nn.Linear(self.y_um_layer_hidden[i], self.y_um_layer_hidden[i+1]) for i in range(self.y_um_num_layers)])
            self.y_um_layer_norm = nn.ModuleList([nn.LayerNorm(self.y_um_layer_hidden[i]) for i in range(self.y_um_num_layers)])
        
        self.y_um_linear_query = nn.Linear(self.user_sequence_model.item_embeddings.embedding_dim, self.intent_embedding.shape[1])
        self.y_um_linear_key = nn.Linear(self.intent_embedding.shape[1], self.intent_embedding.shape[1])
        self.y_um_linear_value = nn.Linear(self.intent_embedding.shape[1], self.intent_embedding.shape[1])
        
        self.linear_intent = nn.Linear(self.intent_embedding.shape[1], self.intent_embedding.shape[1])
        
    # p(m|u,x)
    def distribution_m_ux(self, user_seqs, user_seq_attention_mask, text_embedding):
        user_embedding = self.user_sequence_model(user_seqs, user_seq_attention_mask)[:, -1, :] # [batch_size, user_embedding_dim]
        if self.config.prior.m_ux.use_text_projection:
            text_embedding = self.m_ux_projection(text_embedding)
        if self.m_ux_agg_type == "concat":
            h = torch.cat([user_embedding, text_embedding], dim=1) # [batch_size, user_embedding_dim + intent_hidden]
            for i in range(self.m_ux_num_layers):
                h = F.relu(self.m_ux_agg_layer[i](h))
                # h = F.dropout(h, p=self.m_ux_hidden_dropout_prob, training=self.training)
                if self.m_ux_layer_norm is True and i != self.m_ux_num_layers - 1:
                    h = self.m_ux_layer_norm[i](h)
        elif self.m_ux_agg_type == "gated":
            query = self.m_ux_linear_query(text_embedding) # [batch_size, intent_hidden]
            key = self.m_ux_linear_key(user_embedding) # [batch_size, intent_hidden]
            query = query.unsqueeze(1) # [batch_size, 1, intent_hidden]
            key = key.unsqueeze(2) # [batch_size, intent_hidden, 1]
            attention_weights = torch.matmul(query, key).squeeze() # [batch_size]
            attention_weights = F.softmax(attention_weights/math.sqrt(self.intent_embedding.shape[1]), dim=-1) # [batch_size]
            attention_weights = attention_weights.unsqueeze(1) # [batch_size, 1]
            user_embedding_representation = attention_weights * user_embedding # [batch_size, intent_hidden]
            
            query_2 = self.m_ux_linear_query_2(user_embedding) # [batch_size, intent_hidden]
            key_2 = self.m_ux_linear_key_2(text_embedding) # [batch_size, intent_hidden]
            query_2 = query_2.unsqueeze(1) # [batch_size, 1, intent_hidden]
            key_2 = key_2.unsqueeze(2) # [batch_size, intent_hidden, 1]
            attention_weights_2 = torch.matmul(query_2, key_2).squeeze() # [batch_size]
            attention_weights_2 = F.softmax(attention_weights_2/math.sqrt(self.intent_embedding.shape[1]), dim=-1) # [batch_size]
            attention_weights_2 = attention_weights_2.unsqueeze(1) # [batch_size, 1]
            user_embedding_representation_2 = attention_weights_2 * text_embedding # [batch_size, intent_hidden]
            
            h = user_embedding_representation + user_embedding_representation_2 # [batch_size, intent_hidden]
            
            for i in range(self.m_ux_num_layers):
                h = F.relu(self.m_ux_agg_layer[i](h))
                # h = F.dropout(h, p=self.m_ux_hidden_dropout_prob, training=self.training)
                if self.m_ux_layer_norm is True and i != self.m_ux_num_layers - 1:
                    h = self.m_ux_layer_norm[i](h)
        if self.config.prior.m_ux.is_softmax:
            intent_distribution = F.softmax(torch.matmul(h, self.intent_embedding.t()), dim=-1)
        else:
            intent_distribution = torch.matmul(h, self.intent_embedding.t())
        return intent_distribution # [batch_size, num_intent]
    
    def get_y_mux(self, user_seqs, user_seq_attention_mask, text_embedding, pos_neg_items):
        v_embed = self.item_embedding(pos_neg_items)# [batch_size, num_items, item_embedding_dim]
        m_embed = self.intent_embedding # [num_intent, intent_hidden]
        s_embed = self.user_sequence_model(user_seqs, user_seq_attention_mask)[:, -1, :] # [batch_size, user_embedding_dim]
        x_embed = text_embedding # [batch_size, text_embedding_dim]
        m_ux = self.distribution_m_ux(user_seqs, user_seq_attention_mask, text_embedding) # [batch_size, num_intent]
        y_m = self.get_y_m(pos_neg_items)  # [batch_size, num_intent, num_items]
        m_ux = m_ux.unsqueeze(2) # [batch_size, num_intent, 1]
        y_mux = y_m * m_ux # [batch_size, num_intent, num_items]
        return y_mux
        
    
    def get_y_m(self, target_items, intent_distribution=None):
        # when intent_distribution is not None, it means the inference step, when it is None, it means the ELBO step
        target_items_embedding = self.item_embedding(target_items) # [batch_size, num_items, item_embedding_dim]
        intent_embeddings = self.linear_intent(self.intent_embedding) # [num_intent, intent_hidden]
        intent_embeddings = intent_embeddings.unsqueeze(0).expand(target_items.shape[0], -1, -1) # [batch_size, num_intent, intent_hidden]
        intent_scores =  torch.matmul(intent_embeddings , target_items_embedding.transpose(1, 2)) # [batch_size, num_intent, num_items]
        # intent_scores = F.softmax(intent_scores, dim=1) # [batch_size, num_intent, num_items]
        if intent_distribution is not None:
            if len(intent_distribution.shape) == 2:
                intent_distribution = intent_distribution.unsqueeze(1) # [batch_size, 1, num_intent]
            results = torch.matmul(intent_distribution, intent_scores).squeeze(1) # [batch_size, num_items]
        else:
            results = intent_scores # [batch_size, num_intent, num_items]
        return results
    
    def get_m_yux(self, user_seqs, user_seq_attention_mask, text_embedding, target_items):
        target_items = target_items.unsqueeze(1) # [batch_size, 1]
        py_m = self.get_y_m(target_items).squeeze() # [batch_size, num_intent]
        # py_m = py_m/torch.sum(py_m, dim=-1, keepdim=True)
        py_m = F.softmax(py_m, dim=-1)
        pm_ux = self.distribution_m_ux(user_seqs, user_seq_attention_mask, text_embedding) # [batch_size, num_intent]
        
        scores = py_m * pm_ux
        m_yux = F.softmax(scores, dim=-1)
        # m_yux = scores/torch.sum(scores, dim=-1, keepdim=True)
        return m_yux
    
    # p(y|u,m)
    def distribution_y_um(self, user_seqs, user_seq_attention_mask, intent_distribution, train_steps:str):
        user_embedding = self.user_sequence_model(user_seqs, user_seq_attention_mask)[:, -1, :] # [batch_size, user_embedding_dim]
        if train_steps == "InferenceStep":
            intent_embedding = torch.matmul(intent_distribution, self.intent_embedding)  # [batch_size, intent_hidden]
        elif train_steps == "ELBOStep":
            intent_embedding = self.intent_embedding  # [num_intent, intent_hidden]
            intent_embedding = intent_embedding.unsqueeze(0).expand(user_embedding.size(0), -1, -1)  # [batch_size, num_intent, intent_hidden]
            user_embedding = user_embedding.unsqueeze(1).expand(-1, intent_embedding.size(1), -1)  # [batch_size, num_intent, user_embedding_dim]
        if self.y_um_agg_type == "concat":
            h = torch.cat([user_embedding, intent_embedding], dim=-1)
            for i in range(self.y_um_num_layers):
                if self.y_um_layer_norm is True:
                    h = F.relu(self.y_um_layer_norm[i](self.y_um_agg_layer[i](h)))
                else:
                    h = F.relu(self.y_um_agg_layer[i](h))
                if i != self.y_um_num_layers - 1:
                    h = F.dropout(h, p=self.y_um_hidden_dropout_prob, training=self.training)
        elif self.y_um_agg_type == "gated":
            intent_embedding = self.projection_intent2item(intent_embedding)
            combined = torch.cat([user_embedding, intent_embedding], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            h = gate * user_embedding + (1 - gate) * intent_embedding
            for i in range(self.y_um_num_layers):
                if self.y_um_layer_norm is True:
                    h = F.relu(self.y_um_layer_norm[i](self.y_um_agg_layer[i](h)))
                else:
                    h = F.relu(self.y_um_agg_layer[i](h))
                if i != self.y_um_num_layers - 1:
                    h = F.dropout(h, p=self.y_um_hidden_dropout_prob, training=self.training)
        return h # InferenceStep: [batch_size, item_embedding_dim] or ELBOStep: [batch_size, num_intent, item_embedding_dim]
    
    # EStep, do not train, compute the expectation of intent_distribution
    def prior_network_step(self, user_seqs, user_seq_attention_mask, text_embedding, target_item):
        logp_m_ux = self.distribution_m_ux(user_seqs, user_seq_attention_mask, text_embedding)  # [batch_size, num_intent]
        dis_y_um = self.distribution_y_um(user_seqs, user_seq_attention_mask, logp_m_ux, "ELBOStep")  # [batch_size, num_intent, item_embedding_dim] or [batch_size, item_embedding_dim*2]
        gold_item_embedding = self.item_embedding(target_item)  # [batch_size, 1, item_embedding_dim] or [batch_size, num_items, item_embedding_dim]
        if self.config.prior.y_um.e_step_distribution == "gaussian":
            gaussian_mean, gaussian_logvar = dis_y_um.chunk(2, dim=2) # [batch_size, num_intent, item_embedding_dim]
            logvar_regular = torch.pow(gaussian_logvar, 2).sum(dim=-1).mean(dim=-1) # [batch_size]
            gaussian_mean, gaussian_logvar = gaussian_mean.unsqueeze(1), gaussian_logvar.unsqueeze(1) # [batch_size, 1, num_intent, item_embedding_dim]
            gold_item_embedding = gold_item_embedding.unsqueeze(2).expand(-1, -1, gaussian_mean.shape[2], -1) # [batch_size, num_items, num_intent, item_embedding_dim]
            gaussian_mean = gaussian_mean.expand(-1, gold_item_embedding.shape[1], -1, -1) # [batch_size, num_items, num_intent, item_embedding_dim]
            gaussian_logvar = gaussian_logvar.expand(-1, gold_item_embedding.shape[1], -1, -1) # [batch_size, num_items, num_intent, item_embedding_dim]
            logp_y_um = -0.5 * (
                torch.log(2 * torch.pi * torch.exp(gaussian_logvar)) +
                ((gold_item_embedding - gaussian_mean) ** 2) / torch.exp(gaussian_logvar)
            )  # [batch_size, num_items, num_intent, item_embedding_dim]
            logp_y_um = logp_y_um.sum(dim=-1)  # [batch_size, num_items, num_intent]
        elif self.config.prior.y_um.e_step_distribution == "softmax":
            gold_item_embedding = gold_item_embedding.unsqueeze(2)  # [batch_size, num_items, 1, item_embedding_dim]
            dis_y_um = dis_y_um.unsqueeze(1) # [batch_size, 1, num_intent, item_embedding_dim]
            logp_y_um = F.log_softmax(torch.matmul(gold_item_embedding, dis_y_um.transpose(2, 3)), dim=1)  # [batch_size, num_items, 1, num_intent]
            logp_y_um = logp_y_um.squeeze(2)  # [batch_size, num_items, num_intent]
            logvar_regular = None
        return logp_m_ux, logp_y_um, logvar_regular
    
    def item_embedding_predict_step(self, user_seqs, user_seqs_attention_mask, text_embedding):
        intent_distribution = self.distribution_m_ux(user_seqs, user_seqs_attention_mask, text_embedding) # [batch_size, num_intent]
        if self.config.prior.m_ux.is_softmax:
            intent_distribution = torch.exp(intent_distribution)
        else:
            intent_distribution = F.softmax(intent_distribution, dim=-1)
        pred_item_embedding = self.distribution_y_um(user_seqs, user_seqs_attention_mask, intent_distribution, "InferenceStep") # [batch_size, item_embedding_dim]
        if self.config.prior.y_um.e_step_distribution == "gaussian":
            pred_item_embedding, pred_item_logvar = pred_item_embedding.chunk(2, dim=-1)
        return pred_item_embedding
    
    def get_user_seq_predict(self, user_seqs, user_seqs_attention_mask=None, target_items=None):
        user_embeddings = self.user_sequence_model(user_seqs, user_seqs_attention_mask)[:, -1, :]
        target_items_embedding = self.user_sequence_model.item_embeddings(target_items) # [batch_size, num_items, item_embedding_dim]
        user_embeddings = user_embeddings.unsqueeze(1) # [batch_size, 1, user_embedding_dim]
        scores = torch.matmul(user_embeddings, target_items_embedding.transpose(1, 2)) # [batch_size, 1, num_items]
        scores = scores.squeeze(1) # [batch_size, num_items]
        return scores
    
# Posterior Network q(m|u)
class PosteriorNetwork(nn.Module):
    def __init__(self, config, intent_embedding:torch.Tensor, sas_rec_model:SASRecModel):
        super(PosteriorNetwork, self).__init__()
        self.intent_embedding = nn.Parameter(intent_embedding)
        if config.posterior.use_pretrained_intent:
            print("Use Pretrained seqrec model for Posterior Network")
            self.user_sequence_model = sas_rec_model
        else:
            self.user_sequence_model = SASRecModel(config.precalculated_recmodel)
        self.projection_user2intent = nn.Linear(self.user_sequence_model.item_embeddings.embedding_dim, self.intent_embedding.shape[1])
        self.linear_query = nn.Linear(self.user_sequence_model.item_embeddings.embedding_dim, self.intent_embedding.shape[1])
        self.linear_key = nn.Linear(self.intent_embedding.shape[1], self.intent_embedding.shape[1])
        self.linear_value = nn.Linear(self.intent_embedding.shape[1], self.intent_embedding.shape[1])
        self.config = config
    
    def forward(self, user_seqs, user_seqs_attention_mask=None):
        return self.user_sequence_model(user_seqs, user_seqs_attention_mask)
    
    def euclidean_distance_identity(self,X, A):
        # X: (batch_size, dim)
        # A: (num_anchors, dim)
        
        X_squared = torch.sum(X ** 2, dim=1, keepdim=True)  # (batch_size, 1)
        A_squared = torch.sum(A ** 2, dim=1).unsqueeze(0)  # (1, num_anchors)
        cross_term = torch.matmul(X, A.t())  # (batch_size, num_anchors)
        distances = X_squared + A_squared - 2 * cross_term  # (batch_size, num_anchors)
        return distances
    
    def distribution_m_u(self, user_seqs, user_seqs_attention_mask=None):
        user_embeddings = self.user_sequence_model(user_seqs, user_seqs_attention_mask)[:, -1, :] # [batch_size, user_embedding_dim]
        query = self.linear_query(user_embeddings) # [batch_size, intent_hidden]
        key = self.linear_key(self.intent_embedding) # [num_intent, intent_hidden]
        value = self.linear_value(self.intent_embedding) # [num_intent, intent_hidden]
        key = key.unsqueeze(0).expand(query.shape[0], -1, -1) # [batch_size, num_intent, intent_hidden]
        value = value.unsqueeze(0).expand(query.shape[0], -1, -1) # [batch_size, num_intent, intent_hidden]
        if len(query.shape) == 2:
            query = query.unsqueeze(1) # [batch_size, 1, intent_hidden]
        attention_weights = torch.matmul(query, key.transpose(1, 2)) # [batch_size, 1, num_intent]
        attention_weights = F.softmax(attention_weights/math.sqrt(self.intent_embedding.shape[1]), dim=-1) # [batch_size, 1, num_intent]
        attention_weights = attention_weights.squeeze(1) # [batch_size, num_intent]
        return attention_weights
    
    def get_predict(self, user_seqs, user_seqs_attention_mask=None, target_items=None):
        user_embeddings = self.user_sequence_model(user_seqs, user_seqs_attention_mask) # [batch_size, seq_len, user_embedding_dim]
        query = self.linear_query(user_embeddings) # [batch_size, seq_len, intent_hidden]
        key = self.linear_key(self.intent_embedding) # [num_intent, intent_hidden]
        value = self.linear_value(self.intent_embedding) # [num_intent, intent_hidden]
        # value = self.intent_embedding
        key = key.unsqueeze(0).expand(query.shape[0], -1, -1) # [batch_size, num_intent, intent_hidden]
        value = value.unsqueeze(0).expand(query.shape[0], -1, -1) # [batch_size, num_intent, intent_hidden]
        attention_weights = torch.matmul(query, key.transpose(1, 2)) # [batch_size, seq_len, num_intent]
        attention_weights = F.softmax(attention_weights/math.sqrt(self.intent_embedding.shape[1]), dim=-1) # [batch_size, seq_len, num_intent]
        results_embeddings = torch.matmul(attention_weights, value) # [batch_size, seq_len, intent_hidden]
        reconstruct_loss = F.mse_loss(results_embeddings, user_embeddings)
        
        results_embeddings = results_embeddings[:,-1,:]
        results_embeddings = results_embeddings.unsqueeze(1)
        target_items = self.user_sequence_model.item_embeddings(target_items) # [batch_size, num_items, item_embedding_dim]
        scores = torch.matmul(results_embeddings, target_items.transpose(1, 2)) # [batch_size, 1, num_items]
        scores = scores.squeeze(1) # [batch_size, num_items]
        return scores, reconstruct_loss
    
    def get_user_seq_predict(self, user_seqs, user_seqs_attention_mask=None, target_items=None):
        user_embeddings = self.user_sequence_model(user_seqs, user_seqs_attention_mask)[:, -1, :]
        target_items_embedding = self.user_sequence_model.item_embeddings(target_items) # [batch_size, num_items, item_embedding_dim]
        user_embeddings = user_embeddings.unsqueeze(1) # [batch_size, 1, user_embedding_dim]
        scores = torch.matmul(user_embeddings, target_items_embedding.transpose(1, 2)) # [batch_size, 1, num_items]
        scores = scores.squeeze(1) # [batch_size, num_items]
        return scores
        
# Variational Intent Model, no EM, directly concat for recommendation
class TextIntentModel(nn.Module):
    def __init__(self, config):
        super(TextIntentModel, self).__init__()
        self.config = config
        self.text_projection = nn.Linear(config.text_embedding_dim, config.intent_hidden)
        self.user_projection = nn.Linear(config.precalculated_recmodel.hidden_size, config.intent_hidden)
        
        self.agg_type = config.agg_type
        self.layer_hidden = config.layer_hidden
        self.num_layers = len(self.layer_hidden)
        if self.agg_type == "concat":
            self.layer_hidden = [config.text_embedding_dim + config.precalculated_recmodel.hidden_size] + self.layer_hidden
            self.agg_layer = nn.ModuleList([nn.Linear(self.layer_hidden[i], self.layer_hidden[i+1]) for i in range(self.num_layers)])
        elif self.agg_type == "gated": 
            self.projection_text2user = nn.Linear(config.text_embedding_dim, config.precalculated_recmodel.hidden_size)
            self.gate_layer = nn.Linear(config.precalculated_recmodel.hidden_size * 2, config.intent_hidden)
            self.layer_hidden = [config.intent_hidden] + self.layer_hidden
            self.agg_layer = nn.ModuleList([nn.Linear(self.layer_hidden[i], self.layer_hidden[i+1]) for i in range(self.num_layers)])
    
    def forward(self, user_embedding, text_embedding):
        if self.agg_type == "concat":
            combined = torch.cat([user_embedding, text_embedding], dim=1)
            for i in range(self.num_layers):
                combined = F.relu(self.agg_layer[i](combined))
            return combined
        elif self.agg_type == "gated":
            text_embedding = self.projection_text2user(text_embedding)
            combined = torch.cat([user_embedding, text_embedding], dim=1)
            gate = torch.sigmoid(self.gate_layer(combined))
            h = gate * user_embedding + (1 - gate) * text_embedding
            for i in range(self.num_layers):
                h = F.relu(self.agg_layer[i](h))
            return h