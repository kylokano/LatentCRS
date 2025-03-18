import os
import json
import time
import torch
import transformers

# from fastchat.conversation import get_default_conv_template, compute_skip_echo_len
# from fastchat.serve.inference import load_model
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# from easydict import EasyDict as edict
# from jsonargparse import CLI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
# from vllm import LLM, SamplingParams
import argparse
import copy

from gemini_api import GeminiChat,create_batch_prediction_job
from batch_api import process_batch_prediction
from get_ICL_recommendation import load_model, get_recommendation

prompt_template = """You are a conversational **item** recommendation assistant. Your task is to help a user find the **item**s that meet their requirements. In each turn of the conversation, the user may share two types of information: 1. Hard descriptions: Mandatory requirements that the **item** must fulfill (for example, minimum specifications or key features). 2. Soft descriptions: Additional preferences that are not strictly required.

You have the following candidate item features: **item_feature**.

You should analyze the entire conversation history, to identify: The user's mandatory requirements from the listed item features. And the user's broader preferences about the **item**.

Summarize these in the JSON format below. Use "item features" for hard descriptions that appear in the conversation, and "human intents" for soft descriptions or general preferences. Only list features the user explicitly mentioned and ensure high confidence in their accuracy. Please only return the JSON format.
{
  "human intents": human intents,  
  "item features": item features  
}
conversation_history: **conversation_history**
"""

def get_rs_prompt(datasets_name):
    if datasets_name=="ml-1m":
        item = "movie"
        item_features = {'Film-Noir', 'Musical', 'Fantasy', 'Comedy', 'Adventure', 'Mystery', 
                        'Thriller', 'Western', 'Drama', 'Documentary', 'Romance', 'Horror', 
                        'Sci-Fi', 'Crime', 'War', "Children's", 'Animation', 'Action'}
    elif datasets_name=="cds":
        item = "cd" 
        item_features = {'Jazz', 'Rock', 'Classical', 'Pop', 'Blues', # ... abbreviated feature list
                        'World Music', 'Folk', 'Country', 'Electronic'}
    elif datasets_name=="VideoGames":
        item = "video game"
        item_features = {'Action', 'Adventure', 'RPG', 'Strategy', # ... abbreviated feature list
                        'Sports', 'Simulation', 'Puzzle'}
    return prompt_template.replace("**item**", item).replace("**item_feature**", ", ".join(item_features))

def convert_dialogue2prompt(dialogue):
    dialogue_str = ""
    for turn in dialogue:
        dialogue_str += turn["role"] + ": " + turn["parts"][0]["text"] + "\n"
    return dialogue_str

def get_embedding_input(model_output_list, user_response_list):
    embedding_input_list, item_features_list = [], []
    for model_output, user_response in zip(model_output_list, user_response_list):
        try:
            if model_output.startswith("```"):
                model_output = model_output.replace("```json", "").replace("```", "")
            model_output = json.loads(model_output)
            if len(model_output["human intents"])>0:
                embedding_input = ". ".join(model_output["human intents"])
            else:
                embedding_input = user_response
            # embedding_input = user_response
            item_featuers = model_output["item features"]
        except KeyError:
            print("--------------------------------")
            print(model_output)
            print("--------------------------------")
            embedding_input = user_response
            item_featuers = []
        except Exception as e:
            print(e)
            print(model_output)
            embedding_input = user_response
            item_featuers = []
        embedding_input_list.append(embedding_input)
        item_features_list.append(item_featuers)
    return embedding_input_list, item_features_list

def get_embedding_input_without_filter(user_response_list):
    embedding_input_list, item_features_list = [], []
    for user_response in user_response_list:
        embedding_input = user_response
        item_featuers = []
        embedding_input_list.append(embedding_input)
        item_features_list.append(item_featuers)
    return embedding_input_list, item_features_list

def get_item_str(recommend_item_list, item_features_list):
    output_str = []
    for each_line, item_features in zip(recommend_item_list, item_features_list):
        if len(item_features) > 0 and len(item_features) < 5:
            tmp_str = "We infer you like :" + ", ".join(item_features) + ". The recommended items are: "
        else:
            tmp_str = "The recommended items are: "
        for idx in range(len(each_line)):
            if type(each_line[idx]) == str:
                tmp_str += str(idx+1) + " " + each_line[idx] + " "
            else:
                print(each_line[idx])
                output_str+= str(idx+1) + " tmp_str"
                # tmp_str += str(idx+1) + " " + each_line[idx]["title"] + " "
        output_str.append(tmp_str)
    return output_str

def process_for_batch_prediction(rs_output_lines):
    gemini_input_lines = []
    for data in rs_output_lines:
        if data["success"] is not False:
            continue
        user_dialog = [
        {
            "role": "user",
            "parts": [{"text":data["prompt"]}]
        }]
        for turn in data["llama3_dialogue_history"]:
            turn["role"] = "user" if turn["role"] == "assistant" else "model"
            user_dialog.append(turn)
        gemini_base_template = {
            "customer_id": data["customer_id"],
            "request":{
                "contents": user_dialog,
                "safetySettings": [
                                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}]
                }
        }
        gemini_input_lines.append(gemini_base_template)
    return gemini_input_lines

def get_gemini_output(gemini_input_path, rs_output_lines):
    gemini_output_lines, spcial_id2response = [], {}
    gemini_output_path = gemini_input_path.replace(".jsonl", "_output.jsonl")
    with open(gemini_output_path, "r", encoding='utf-8') as fr:
        output = [json.loads(line) for line in fr.readlines()]
        for data in output:
            spcial_id2response[data["customer_id"]] = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
    for data in rs_output_lines:
        if data["success"] is not False:
            gemini_output_lines.append(data)
        else:
            data["response"] = spcial_id2response[data["customer_id"]]
            gemini_output_lines.append(data)
    return gemini_output_lines

def main(from_json: str = None,
         model=None,
         tokenizer=None,
         sampling_params=None,
         dataset_name="cds",
         turn_num=0,
         batch_size=10,
         given_topK=5,
         do_gemini=True,
         rec_config_path=None,
         prior_model=None,
         item_size=None):
    """
    Main function for multi-turn recommendation
    
    Args:
        from_json: Input json file path
        model: Language model
        tokenizer: Tokenizer
        sampling_params: Generation parameters
        dataset_name: Name of dataset
        turn_num: Current conversation turn
        batch_size: Batch size for generation
        given_topK: Number of items to recommend
        do_gemini: Whether to use Gemini model
        rec_config_path: Path to recommender config
        prior_model: Pre-trained recommendation model
        item_size: Size of item set
    """
    
    PREFIXTOKEN = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    ENDTOKEN = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    print("open turn",turn_num)
    print("Load prompt from ",from_json.replace(".jsonl",f"_multi_turn_batch_turn_{turn_num}_user_similator.jsonl"))
    with open(from_json.replace(".jsonl",f"_multi_turn_batch_turn_{turn_num}_user_similator.jsonl"), "r", encoding='utf-8') as fr:
        lines = [json.loads(line) for line in fr.readlines()] # customer_id, history, target, prompt, response
    
    # 新增一个llama3_dialogue_history的模块
    # 处理最开始的状态
    
    if "llama3_dialogue_history" not in lines[0]:
        for data in lines:
            data["llama3_dialogue_history"]=[{
            "role": "user",
            "parts": [{"text":data["response"]}]
            }]
            data["success"] = False
    else:
        for data in lines:
            data["llama3_dialogue_history"].append({
            "role": "user",
            "parts": [{"text":data["response"]}]
            })
    if "already_recommend" not in lines[0]:
        for data in lines:
            data["already_recommend"] = []
    print('using ICL (Llama3) inference mode')
    # 收集所有需要生成的prompt
    prompts_to_generate, last_user_response = [], []
    for data in lines:
        
        # 处理每个会话的轮数
        if "item_features" not in data:
            data["item_features"] = []
        last_user_response.append(data["llama3_dialogue_history"][-1]["parts"][0]["text"])
        dialogue_str = convert_dialogue2prompt(data["llama3_dialogue_history"])
        llama3_prompt = get_rs_prompt(dataset_name).replace("**conversation_history**", dialogue_str)
        # data["current_turn_prompt"] = llama3_prompt
        # 生成推荐
        prompt = llama3_prompt
        dialog = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_to_generate.append(prompt)

    llama3_output_text, llama3_output_embeddings = [],[]
    tmp_item_featuers = []
    batch_size_num = len(prompts_to_generate)//batch_size + 1
    for i in tqdm(range(batch_size_num),desc="Generating Llama3 RS returns"):
        if i*batch_size+batch_size>len(prompts_to_generate):
            model_inputs = tokenizer(prompts_to_generate[i*batch_size:], return_tensors="pt", padding=True, padding_side="left").to("cuda")
            batch_user_response = last_user_response[i*batch_size:]
        else:
            model_inputs = tokenizer(prompts_to_generate[i*batch_size:(i+1)*batch_size], return_tensors="pt", padding=True, padding_side="left").to("cuda")
            batch_user_response = last_user_response[i*batch_size:(i+1)*batch_size]
        outputs = model.generate(**model_inputs, generation_config=sampling_params, pad_token_id=tokenizer.eos_token_id)
        input_length = model_inputs.input_ids.shape[1]
        outputs = outputs[:,input_length:]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        llama3_output_text.extend(outputs)
        
        embedding_input_list, item_features_list = get_embedding_input(outputs, batch_user_response)
        tmp_item_featuers.extend(item_features_list)
        
        outputs_embedding = [PREFIXTOKEN+"This sentence : \"*sent 0*\" means in one word:\"".replace("*sent 0*", embedding_input)+ENDTOKEN for embedding_input in embedding_input_list]
        outputs_embedding = tokenizer(outputs_embedding, return_tensors="pt", padding=True, padding_side="left").to("cuda")
        outputs_embedding = model.base_model(**outputs_embedding, return_dict=True)
        outputs_embedding = outputs_embedding.last_hidden_state
        outputs_embedding = outputs_embedding[:,-1,:]
        llama3_output_embeddings.append(outputs_embedding.to("cpu").detach())
    
    print("Saving Intermediate results")
    dataset_path = "results/ours-llama3/mediate/"+from_json.split("/")[-1].replace(".jsonl","_multi_turn_batch_turn_"+str(turn_num)+"_recommend.jsonl")
    precal_embedding_path = "results/ours-llama3/mediate/"+from_json.split("/")[-1].replace(".jsonl","_multi_turn_batch_turn_"+str(turn_num)+"_embeddings.pt")
    item_features_constraint, already_recommend = [], []
    with open(dataset_path, "w", encoding='utf-8') as fw:
        output_lines = copy.deepcopy(lines)
        special_ids = []
        for i in range(len(output_lines)):
            output_lines[i]["prompt"] = prompts_to_generate[i]
            output_lines[i]["response"] = llama3_output_text[i]
            output_lines[i]["item_features"] = list(set(output_lines[i]["item_features"]) | set(tmp_item_featuers[i]))
            output_lines[i]["item_features"] = tmp_item_featuers[i]
            item_features_constraint.append(output_lines[i]["item_features"])
            already_recommend.append(output_lines[i]["already_recommend"])
            special_ids.append(output_lines[i]["customer_id"])
            fw.write(json.dumps(output_lines[i]) + '\n')
    with open(precal_embedding_path, "wb") as fw:
        llama3_output_embeddings = torch.cat(llama3_output_embeddings, dim=0)
        saved_output = {"special_id":special_ids,
                        "embeddings":llama3_output_embeddings}
        torch.save(saved_output, fw)
    
    # 用于单独测试recommend模块
    # item_features_constraint, already_recommend = [], []
    # with open(dataset_path, "r", encoding='utf-8') as fr:
    #     lines = [json.loads(line) for line in fr.readlines()]
    #     for line in lines:
    #         if "already_recommend" not in line:
    #             line["already_recommend"] = []
    #         item_features_constraint.append(line["item_features"])
    #         already_recommend.append(line["already_recommend"])
    
    pred_itemids, recommend_items, recommend_item_features, is_success = get_recommendation(dataset_name, rec_config_path, dataset_path, precal_embedding_path, id2modelid, prior_model, item_size, given_topK=5, item_features_constraint=item_features_constraint, already_recommend=already_recommend)
    
    # print("success rate", sum(is_success)/len(is_success))
    recommended_item_str = get_item_str(recommend_items, item_features_constraint)
    rs_output_lines = copy.deepcopy(lines)
    
    for i in range(len(rs_output_lines)):
        rs_output_lines[i]["already_recommend"].extend(pred_itemids[i])
        rs_output_lines[i]["llama3_dialogue_history"].append({
            "role": "assistant",
            "parts": [{"text":recommended_item_str[i]}]
        })
        rs_output_lines[i]["item_features"] = item_features_constraint[i]
        if is_success[i] == 1 and rs_output_lines[i]["success"] is False:
            rs_output_lines[i]["success"] = turn_num + 1
    
    with open(from_json.replace(".jsonl",f"_multi_turn_batch_turn_{turn_num}_rs_output.jsonl"), "w", encoding='utf-8') as fw:
        for data in rs_output_lines:
            fw.write(json.dumps(data) + '\n')
    
    gemini_input_lines = process_for_batch_prediction(copy.deepcopy(rs_output_lines))
    gemini_input_path = "results/ours-llama3/gemini_user_similator/"+"ICL"+from_json.split("/")[-1].replace(".jsonl",f"_multi_turn_batch_turn_{turn_num}_gemini_input.jsonl")
    with open(gemini_input_path, "w", encoding='utf-8') as fw:
        for data in gemini_input_lines:
            fw.write(json.dumps(data) + '\n')
    turn_num+=1
    if turn_num<5 and do_gemini:
        process_batch_prediction(gemini_input_path)
        user_similator_output = get_gemini_output(gemini_input_path, rs_output_lines)
        with open(from_json.replace(".jsonl",f"_multi_turn_batch_turn_{turn_num}_user_similator.jsonl"), "w", encoding='utf-8') as fw:
            for data in user_similator_output:
                fw.write(json.dumps(data) + '\n')
    else:
        success_rate, average_turn_num = 0, 0
        for data in rs_output_lines:
            if data["success"] is not False:
                success_rate+=1
                average_turn_num+=data["success"]
            else:
                average_turn_num+=5
        print("success rate", success_rate/len(rs_output_lines))
        print("average turn num", average_turn_num/len(rs_output_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_json', type=str, default='results/sample_data.jsonl')
    parser.add_argument('--dataset_name', type=str, default='cds')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='path/to/model')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--n_print', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--use_tqdm', type=bool, default=True)
    parser.add_argument('--given_topK', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--recommend_model_path', type=str, default='path/to/model.pth')
    parser.add_argument('--recommend_model_config_path', type=str, default='path/to/config.json')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    llm_model.requires_grad = False
    # model = LLM(model=args.pretrained_model_name_or_path, trust_remote_code=True, tensor_parallel_size=args.GPU_NUM,dtype=torch.bfloat16, enforce_eager=True,max_model_len=4096,  # 添加这行
    #         gpu_memory_utilization=0.9)
    sampling_params = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        # temperature=args.temperature,  # 对于问答任务，建议使用较低的温度
        top_p=1.0,
    )
    
    id2modelid, recommend_model, item_size = load_model(args.dataset_name, args.recommend_model_config_path, args.recommend_model_path)
    recommend_model.requires_grad = False
    recommend_model.to("cuda")
    
    for turn_num in range(5):
        main(from_json=args.from_json,
            model=llm_model,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            dataset_name=args.dataset_name,
            turn_num=turn_num,
            batch_size=args.batch_size,
            given_topK=args.given_topK,
            rec_config_path=args.recommend_model_config_path,
            prior_model=recommend_model,
            item_size=item_size,
            do_gemini=True
            )
        # main_without_filter(from_json=args.from_json.replace(".jsonl",f"_without_filter.jsonl"),
        #     model=llm_model,
        #     tokenizer=tokenizer,
        #     sampling_params=sampling_params,
        #     dataset_name=args.dataset_name,
        #     turn_num=turn_num,
        #     batch_size=args.batch_size,
        #     given_topK=args.given_topK,
        #     rec_config_path=args.recommend_model_config_path,
        #     prior_model=recommend_model,
        #     item_size=item_size
        #     )
