import pandas as pd
import re
import random
import json
import copy
import os
import argparse

USER_SIMILARITOR = "You are a user chatting with a recommender for {item} recommendation in turn. Your history is {history}. Your target items: {target}. Here is the information about target you could use: {target_item_info} \
You must follow the instructions below during chat. \
State your intention to guide the recommender towards the target item. \
If the recommender recommends {target}, you should accept. \
If the recommender recommends other items, you should refuse them and provide the information about {target}. \
You should never directly tell the target item title. \
If the recommender asks for your preference, you should provide the information about {target}. \
You could provide your history. \
You should never directly tell the target item title. \
Your output is only allowed to be the words from the user you act. \
If you think the conversation comes to an ending, output a ⟨END⟩. \
You should never directly tell the target item. Only use the provided information about the target. \
Never give many details about the target items at one time. Less than 3 conditions is better. \
Now lets start, you first, act as a user."
random.seed(2024)


# For validation, use -2 item, for test, use -1 item
def get_item_descriptions(dataset_name:str, data_type:str):
    item_id2descriptions = {}
    if dataset_name == "ml-1m":
        item_features = pd.read_csv("data/ml-1m/movie_features.csv", sep='\t').astype(str)
        # 使用正则表达式提取电影名称和年份
        item_features["Year"] = item_features["Title"].apply(lambda x: re.search(r'\((\d{4})\)', x).group(1))
        item_features["MovieName"] = item_features["Title"].apply(lambda x: re.sub(r'\(\d{4}\)', '', x).strip())

        # 删除原始的 Title 列，并重新排列列顺序
        item_features = item_features[["MovieID", "MovieName", "Year", "Genres"]]
        for item_id, item_names, item_year, item_genres in zip(item_features["MovieID"], item_features["MovieName"], item_features["Year"], item_features["Genres"]):
            item_genres = item_genres.replace("|", ", ")
            if data_type == "all":
                text_description = f"{item_names}, released in {item_year}, and belonging to the genres of {item_genres}."
            elif data_type == "name":
                text_description = f"{item_names}"
            else:
                raise KeyError("Check the data_type")
            item_id2descriptions[item_id] = text_description
    elif dataset_name == "cds":
        item_features = pd.read_csv("data/cds/CDs_features.csv", sep='\t').astype(str)
        # CDsID	rawID	title	main_category	categories	description
        for item_id, item_name, item_categories, item_description in zip(item_features["CDsID"], item_features["title"], item_features["categories"], item_features["description"]):
            item_categories = item_categories.replace("CDs & Vinyl,", "")
            if data_type == "all":
                text_description = f"{item_name}, belongs to the category of {item_categories}, and the description is {item_description}."
            elif data_type == "name":
                text_description = f"{item_name}"
            else:
                raise KeyError("Check the data_type")
            item_id2descriptions[item_id] = text_description
    elif dataset_name == "VedioGames":
        item_features = pd.read_csv("data/VedioGames/VedioGames_features.csv", sep='\t').astype(str)
        # GamesID	rawID	title	main_category	categories	description	features
        for item_id, item_name, item_categories, item_description, item_features in zip(item_features["GamesID"], item_features["title"], item_features["categories"], item_features["description"], item_features["features"]):
            item_categories = item_categories.replace("Video Games,", "")
            if data_type == "all":
                text_description = f"{item_name}, belongs to the category of {item_categories}, and the description is {item_description}."
            elif data_type == "name":
                text_description = f"{item_name}"
            else:
                raise KeyError("Check the data_type")
            item_id2descriptions[item_id] = text_description
    else:
        raise KeyError("Check the dataset_name")
    return item_id2descriptions

# generate data for training, please note here only get the -3 items, leave -2 and -1 for evaluation and testing
def get_user_history(dataset_name:str, max_history:int=25, data_argument_type:str="raw", data_type:str="train"):
    raw_useridhistory = {}
    user_id2history = {}
    if dataset_name == "ml-1m":
        inter_data = pd.read_csv("data/ml-1m/ml-1m.csv", sep='\t', header=None).astype(str)
    elif dataset_name == "cds":
        inter_data = pd.read_csv("data/cds/cds.csv", sep='\t', header=None).astype(str)
    elif dataset_name == "VedioGames":
        inter_data = pd.read_csv("data/VedioGames/VedioGames.csv", sep='\t', header=None).astype(str)
    else:
        raise KeyError("Check the dataset_name")
    for user_id, user_history in zip(inter_data[0], inter_data[1]):
        user_history = user_history.split(" ")
        # For training use -3 items, for validation use -2 items, for testing use -1 items
        if data_type == "train":
            raw_useridhistory[user_id] = user_history[:-2]
        elif data_type == "valid":
            raw_useridhistory[user_id] = user_history[:-1]
        else:
            raw_useridhistory[user_id] = user_history
    if data_argument_type == "raw":
        for user_id,raw_history in raw_useridhistory.items():
            if len(raw_history) >= max_history:
                user_id2history[user_id] = raw_history[-max_history:]
            else:
                user_id2history[user_id] = raw_history
    elif data_argument_type == "selection":
        for user_id, raw_history in raw_useridhistory.items():
            if len(raw_history) < max_history/3:
                continue
            rand_destine = random.randint(max_history//6, len(raw_history) - max_history//6)
            if rand_destine >= max_history:
                user_id2history[user_id] = raw_history[rand_destine - max_history:rand_destine]
            else:
                user_id2history[user_id] = raw_history[:rand_destine]
    return user_id2history

def get_gemini_response(dataset_name:str, data_argument_type:str, saved_path:str, saved_data_name:str, random_seed: str|int, data_type:str="train", max_history:int=25):
    os.makedirs(saved_path, exist_ok=True)
    
    item_id2names = get_item_descriptions(dataset_name, "name")
    item_id2description = get_item_descriptions(dataset_name, "all")
    
    if dataset_name == "ml-1m":
        item_type = "movies"
    elif dataset_name == "cds":
        item_type = "CDs"
    elif dataset_name == "VedioGames":
        item_type = "Video Games"
    else:
        raise KeyError("Check the dataset_name")
    
    user_history = get_user_history(dataset_name, max_history=max_history, data_argument_type=data_argument_type, data_type=data_type)
    writter = open(os.path.join(saved_path, saved_data_name), "w", encoding='utf8')
    gemini_base_template = {
        "customer_id": "",
        "request":{
            "contents": [{"role":"user",
                        "parts": [{"text": ""}]}],
            "safetySettings": [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}]
            }
        }
    for user_id, user_history in user_history.items():
        history, target = user_history[:-1], user_history[-1]
        history_description = "; ".join(map(lambda x: item_id2names[x], history))
        target_name = item_id2names[target]
        target_description = item_id2description[target]
        
        prompt = USER_SIMILARITOR.format(item=item_type, history=history_description, target=target_name, target_item_info=target_description)
        customer_id = str(user_id) + "_" + data_argument_type + "_" + str(random_seed)
        jsonl_text = copy.deepcopy(gemini_base_template)
        jsonl_text["history"] = history
        jsonl_text["target"] = target
        jsonl_text["customer_id"] = customer_id
        jsonl_text["request"]["contents"][0]["parts"][0]["text"] = prompt
        writter.write(json.dumps(jsonl_text) + "\n")     
    writter.close()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--max_history", type=int, default=25)
    args = parser.parse_args()
    
    dataset = args.dataset
    max_history = args.max_history
    data_type_list = ["train", "train_selection", "valid", "test"]
    
    for data_type in data_type_list:
        if data_type == "train_selection":
            data_argument_type = "selection"
            random_seed_list = [1755, 2097, 3068, 6868, 7702]
        elif data_type == "train":
            data_argument_type = "raw"
            random_seed_list = ["raw"]
        else:
            data_argument_type = "raw"
            random_seed_list = [data_type]
        
        for rsd in random_seed_list:
            if type(rsd) == int:
                random.seed(rsd)
            saved_data_name = f"{dataset}_{data_argument_type}_seed{rsd}.jsonl"
            get_gemini_response(dataset, data_argument_type, f"generated_data/{dataset}/prompt4generation", saved_data_name, rsd, data_type, max_history)
        