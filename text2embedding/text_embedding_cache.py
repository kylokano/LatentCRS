from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel
from typing import List
import json
import os
import gc
from tqdm import tqdm
import argparse
import torch
# from my_utils import rank0_print

# 定义自定义数据集
class TextIntentionDataset(Dataset):
    def __init__(self, special_ids: List[str], texts: List[str], tokenizer: AutoTokenizer):
        assert len(special_ids) == len(texts), "Please ensure the length of id match texts"
        self.special_ids = special_ids
        self.texts = texts
        self.tokenizer = tokenizer
        # 对所有文本进行标记化，并记录长度
        print("Data Tokenizer")
        self.encodings = [self.tokenizer(text, return_length=True) for text in texts]
        self.lengths = [enc['length'] for enc in self.encodings]
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return {
            'special_id': self.special_ids[idx],
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'length': self.lengths[idx]
        }

# 自定义批次采样器
class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size):
        self.batch_size = batch_size
        # 按长度排序的索引
        self.sorted_indices = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
        # 将索引分成批次
        self.batches = [self.sorted_indices[i:min(i + batch_size, len(lengths))] for i in range(0, len(self.sorted_indices), batch_size)]
        
        
    def __iter__(self):
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)
    
class SpecialIDFilter(object):
    def __init__(self, special_token):
        if type(special_token) is not str:
            special_token = str(special_token)
        self.special_token = special_token
    
    def __call__(self, ids: str):
        if ids.split("_")[-1] == self.special_token:
            return True
        else:
            return False

# 定义 collate 函数，用于动态 padding
class CollateFN(object):
    def __init__(self, tokenizer:AutoTokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        special_ids = [item['special_id'] for item in batch]
        # 使用 tokenizer 的 pad 方法进行动态 padding
        batch_encoding = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            return_tensors='pt',
            padding_side='left'
        )
        return special_ids, batch_encoding
    


def get_textintention_dataloader(dataset_name:str, tokenizer: AutoTokenizer, sampler_type: str, batch_size:int,  id_filter, collate_fn, task_prompt:str|None = None, data_type:str = "train"):
    assert data_type in ["train", "valid", "test"], "data_type should be in ['train', 'valid', 'test']"
    if data_type == "train":
        dataset_path = f"generated_data/{dataset_name}/processed_data/{dataset_name}_combination.jsonl"
    else:
        dataset_path = f"generated_data/{dataset_name}/processed_data/{dataset_name}_raw_seed{data_type}.jsonl"
    special_ids, texts = [], []
    with open(dataset_path, "r", encoding="utf8") as f:
        for line in f:
            if not line:
                continue
            line_data = json.loads(line)
            if id_filter(line_data["customer_id"]):
                special_ids.append(line_data["customer_id"])
                if task_prompt is None:
                    texts.append(PREFIXTOKEN+line_data["response"]+ENDTOKEN)
                else:
                    texts.append(PREFIXTOKEN+task_prompt.replace("*sent 0*", line_data["response"])+ENDTOKEN)
    print("Text Examples: ", texts[0])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    intention_dataset = TextIntentionDataset(special_ids=special_ids, texts=texts, tokenizer=tokenizer)
    
    if sampler_type == "sorted":
        batch_sampler = SortedBatchSampler(intention_dataset.lengths, batch_size)
    elif sampler_type == "sequential":
        batch_sampler = SequentialSampler(intention_dataset)
    else:
        raise NotImplementedError("Sampler type not implemented, should in ['sorted', 'sequential']")
    
    return DataLoader(intention_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=8)
    


# No need to use, as we do not train LLM
# class SortedBatchSampler(Sampler):
#     def __init__(self, lengths, batch_size, shuffle=True):
#         self.batch_size = batch_size
#         self.lengths = lengths
#         self.shuffle = shuffle
#         self.create_batches()
        
#     def create_batches(self):
#         # 按长度排序的索引
#         self.sorted_indices = sorted(range(len(self.lengths)), key=lambda k: self.lengths[k])
        
#         # 定义分块大小（例如，每个块包含1000个样本）
#         chunk_size = 1000
#         # 将排序后的索引分块
#         chunks = [self.sorted_indices[i:i + chunk_size] for i in range(0, len(self.sorted_indices), chunk_size)]
        
#         # 在每个块内打乱顺序
#         if self.shuffle:
#             for chunk in chunks:
#                 random.shuffle(chunk)
        
#         # 合并所有块的索引
#         shuffled_indices = [idx for chunk in chunks for idx in chunk]
        
#         # 将索引分成批次
#         self.batches = [shuffled_indices[i:i + self.batch_size] for i in range(0, len(shuffled_indices), self.batch_size)]
        
#     def __iter__(self):
#         # 每个 epoch 开始时重新创建批次
#         self.create_batches()
#         for batch in self.batches:
#             yield batch
            
#     def __len__(self):
#         return len(self.batches)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, must in ['ml-1m', 'cds', 'VedioGames']")
    parser.add_argument("--model_path", type=str, required=True, help="The path of the LLMs, here we use llama-3.1-8b-instruct")
    parser.add_argument("--batch_size", type=int, required=True, help="The batch size")
    parser.add_argument("--gpu_id", type=str, required=True, help="The gpu id")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    task_prompts = {"raw": "This sentence : \"*sent 0*\" means in one word:\"",
                #     "Product_Review_Rating":"In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : \"*sent 0*\" reflects the sentiment in one word:\"",
                # "Emotion Detection": "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : \"*sent 0*\" conveys the emotion in one word:\"",
                # "Sentiment_Analysis": "You are presented with a user review. Your task is to identify the sentiment of the review as one of the following: 'Positive,' 'Negative,' or 'Neutral.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Similarity Check": "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                # "Key_Fact_Identification":"In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : \"*sent 0*\" encapsulates the key fact in one word:\"",
                # "Intent_Detection": "You are presented with a user query. Your task is to identify the user's intent as one of the following: 'Information Seeking,' 'Purchase Request,' 'Support Inquiry,' or 'Feedback Submission.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Query_Intention":"In this task, analyze the provided sentence to determine the user's intention. Classify it as either 'Request,' 'Complaint,' 'Suggestion,' or 'Inquiry.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Product_Interest": "In this task, analyze the given review and classify the user's sentiment towards the product as 'Highly Interested,' 'Somewhat Interested,' 'Neutral,' 'Somewhat Disinterested,' or 'Not Interested.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Product_Recommendation": "You are given a product review. Your goal is to predict the user's preference for this product on a scale of 1-5, where 1 indicates 'dislike' and 5 indicates 'like very much.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Content_Based_Recommendations": "You are given a user's product review. Classify the review based on the main attribute highlighted: 'Quality,' 'Price,' 'Usability,' or 'Features.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
                # "Tone_of Product": "In this task, analyze the user review to determine its tone regarding the product's value: 'High Value,' 'Fair Value,' or 'Low Value.' For this task, This sentence: \"*sent 0*\" means in one word:\"",
    }
    for data_type in ["train", "valid", "test"]:
        if data_type == "train":
            filter_token_list = ["raw", 1755, 2097, 3068, 6868,7702]
        elif data_type == "valid":
            filter_token_list = [data_type]
        else:
            filter_token_list = [data_type]
        
        # if data_type != "valid":
        #     continue
        # if data_type == "test":
        #     continue
    
    
        # dataset_name = "ml-1m"
        # dataset_name = "VedioGames"
        dataset_name = args.dataset
        model_path = args.model_path
        
        if "llama-3" in model_path:
            PREFIXTOKEN = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            ENDTOKEN = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            raise NotImplementedError("Model not implemented, Please provide the PREFIXTOKEN and ENDTOKEN")
        batch_size = args.batch_size
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        collate_fn = CollateFN(tokenizer=tokenizer)

        model = AutoModel.from_pretrained(model_path).to("cuda")
        print(f"Model Loaded from {model_path}")
        saved_file_path = f"text2embedding/embedding_cache/{dataset_name}"
        
        idx = 0
        for task_key, task_prompt in task_prompts.items():
            print(f"DataType: {data_type}")
            print(f"Task: {task_key}")
            for filtered_token in filter_token_list:
                saved_file_name = f"{task_key}_{filtered_token}.pt"
                if os.path.exists(os.path.join(saved_file_path, saved_file_name)):
                    print(f"File {saved_file_name} already exists, skip")
                    continue
                
                all_special_ids, all_embeddings = [], []
                id_filter = SpecialIDFilter(filtered_token)
                embedding_dataloader = get_textintention_dataloader(dataset_name=dataset_name, tokenizer=tokenizer, sampler_type="sorted", batch_size=batch_size, id_filter=id_filter, collate_fn=collate_fn, task_prompt=task_prompt, data_type=data_type)
                # 迭代 DataLoader
                for batch in tqdm(embedding_dataloader, desc=f"Task: {task_key}, Filtered Token: {filtered_token}"):
                    special_ids, batch_data = batch
                    batch_data = {k:v.to("cuda") for k,v in batch_data.items()}
                    outputs = model(**batch_data, return_dict=True)
                    # Get the latest hidden states
                    embeddings = outputs.last_hidden_state
                    embeddings = embeddings[:,-1,:]
                    all_embeddings.append(embeddings.to("cpu").detach())
                    all_special_ids = all_special_ids + special_ids
                    
                all_embeddings = torch.cat(all_embeddings, dim=0)
                assert len(all_special_ids) == all_embeddings.shape[0]
                
                if not os.path.exists(saved_file_path):
                    os.mkdir(saved_file_path)
                
                with open(os.path.join(saved_file_path, saved_file_name), "wb") as f:
                    saved_output = {"special_id":all_special_ids,
                                    "embeddings":all_embeddings}
                    torch.save(saved_output, f)
                del all_embeddings
                del all_special_ids
                gc.collect()
                print(f"Saved at {saved_file_name}")
