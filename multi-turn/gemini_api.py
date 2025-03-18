import json
import google.generativeai as genai
from typing import List, Dict, Any
import time
from google.api_core.exceptions import TooManyRequests
import os
import requests
import sys
class GeminiChat:
    def __init__(self, api_key: str,model="gemini-1.5-pro-002",temperature: float = 0,max_output_tokens: int = 1024):
        genai.configure(api_key=api_key,transport="rest")
        proxy_url = "114.212.20.133:808"
    
        # 设置环境变量
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        self.model = genai.GenerativeModel(model)
        generation_config = genai.GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        )
        self.generation_config = generation_config
        self.wait_time=1
        self.last_call_time=time.time()
    def process_conversation(self, 
                           message: List[Dict[str, str]], 
                           ) -> Dict[str, Any]:
        """
        处理对话请求
        messages格式: [
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
            ....
            ]
        """
        # chat = self.model.start_chat(
        # history=message[:-1],
        # )
        # response = chat.send_message(message[-1]["parts"],generation_config=self.generation_config)
        # print(response)
        # return response
        # current_time = time.time()
        # # 检查是否需要等待
        # if current_time - self.last_call_time < self.wait_time:
        #     time.sleep(self.wait_time - (current_time - self.last_call_time))  # 等待剩余时间
        max_retries=100
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    contents=message,
                    generation_config=self.generation_config,
                )
                self.last_call_time=time.time()
                return response._result.candidates[0].content.parts[0].text
            
            except TooManyRequests as e:
                #raise
                print(f"Too many requests. Attempt {attempt + 1} of {max_retries}. Retrying in 60 seconds...")
                time.sleep(60)  # 指数退避
            except Exception as e:
                print(f"An error occurred: {e}")
                break  # 其他错误，退出循环

        print("Max retries exceeded.")
        sys.exit(1) 
        
def create_batch_prediction_job(input_uri, model="publishers/google/models/gemini-1.5-pro-002", location="us-central1", project_id="", ACCESS_TOKEN="" ):
    """
    创建批量预测任务
    
    Args:
        input_uri (str): 输入文件的GCS路径
        model (str): 模型名称
        location (str): 地理位置
        project_id (str): 项目ID
        access_token (str): 访问令牌
    
    Returns:
        dict: 响应结果
    """
    proxy_url = "114.212.20.133:808"
    
    # 设置环境变量
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    ACCESS_TOKEN = ""  # 使用 gcloud auth print-access-token 获取

    output_uri = input_uri.replace(".jsonl", "_output")
    endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs"
    
    request_body = {
        "displayName": "batch_prediction_job",
        "model": model,
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {
                "uris": [input_uri]
            }
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {
                "outputUriPrefix": output_uri
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }

    response = requests.post(endpoint, headers=headers, json=request_body)

    if response.status_code in [200, 202]:
        print("批量预测任务创建成功。")
        print(response.json())
        return response.json()
    else:
        print(f"创建批量预测任务失败: {response.text}")
        return None


# 使用示例

if __name__ == "__main__":
    create_batch_prediction_job("gs://llmrec-data/cds_raw_seedtest_sample_500_llama3_processed_multi_turn_batch_1.jsonl")


