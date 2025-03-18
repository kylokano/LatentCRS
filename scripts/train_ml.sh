export CUDA_VISIBLE_DEVICES=0
learning_rate=0.001
e_negative_loss=5
m_negative_loss=0
m_rec_loss=5
num_intents=512


python main.py exp_name=ml_only_inference_lr${learning_rate}_num_intents${num_intents} datasets=ml-1m datasets.meta_task_keys=[raw] debug=false learning_rate=${learning_rate} scheduler.end_factor=0.01 scheduler.total_iters=50 prior.use_pretrained_intent=true posterior.use_pretrained_intent=true train_preintent=false use_negetive_loss=true e_negetive_loss_weight=5 neg_loss_temp=1 no_mstep=true posterior.load_pretrained=false only_rec_epoch=500 same_prerec_model=true train_preintent=false validation_first=true posterior.is_softmax=true lambda_reconstruct=0 same_y_m=${same_y_m} wandb.project=interrec_final early_stop.stopbywhich=EStep datasets.precalculated_recmodel.kmeans_num_clusters=${num_intents}

python main.py exp_name=ml_emfixe_${learning_rate}_m_rec_loss${m_rec_loss}_intent_num${num_intents}_m_negetive_loss${m_negative_loss} datasets=ml-1m datasets.meta_task_keys=[raw] debug=false learning_rate=${learning_rate} scheduler.end_factor=0.01 scheduler.total_iters=50 prior.use_pretrained_intent=true posterior.use_pretrained_intent=true train_preintent=false use_negetive_loss=true e_negetive_loss_weight=${e_negative_loss} m_negetive_loss_weight=${m_negative_loss} m_rec_loss_weight=${m_rec_loss} neg_loss_temp=1 posterior.load_pretrained=true only_rec_epoch=0 same_prerec_model=true train_preintent=false validation_first=true posterior.is_softmax=true lambda_reconstruct=0 same_y_m=${same_y_m} prior.load_pretrained=false datasets.precalculated_recmodel.kmeans_num_clusters=${num_intents} no_estep=true wandb.project=interrec_test posterior.pretrained_path=./outputs/checkpoints/ml_only_inference_lr${learning_rate}_num_intents${num_intents}_inference.pt

python main.py exp_name=ml_em_allpretrain_${learning_rate}_m_rec_loss${m_rec_loss}_intent_num${num_intents}_m_negetive_loss${m_negative_loss} datasets=ml-1m datasets.meta_task_keys=[raw] debug=false learning_rate=${learning_rate} scheduler.end_factor=0.01 scheduler.total_iters=50 prior.use_pretrained_intent=true posterior.use_pretrained_intent=true train_preintent=false use_negetive_loss=true e_negetive_loss_weight=${e_negative_loss} m_negetive_loss_weight=${m_negative_loss} m_rec_loss_weight=${m_rec_loss} neg_loss_temp=1 posterior.load_pretrained=true only_rec_epoch=0 same_prerec_model=true train_preintent=false validation_first=true posterior.is_softmax=true lambda_reconstruct=0 same_y_m=${same_y_m} prior.load_pretrained=true datasets.precalculated_recmodel.kmeans_num_clusters=${num_intents} no_estep=false wandb.project=interrec_test posterior.pretrained_path=./outputs/checkpoints/ml_only_inference_lr${learning_rate}_num_intents${num_intents}_inference.pt prior.pretrained_path=./outputs/checkpoints/ml_emfixe_${learning_rate}_m_rec_loss${m_rec_loss}_intent_num${num_intents}_m_negetive_loss${m_negative_loss}_prior.pt