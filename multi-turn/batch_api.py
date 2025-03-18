import time
import vertexai

from vertexai.batch_prediction import BatchPredictionJob
import os


from google.cloud import storage
import os

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )

def process_batch_prediction(input_file_path):
    """
    处理批量预测任务
    
    Args:
        input_file_path (str): 输入文件路径
    
    Returns:
        bool: 任务是否成功
    """
    # 初始化配置
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    os.environ["GCLOUD_PROJECT"] = ""
    os.environ["GCLOUD_LOCATION"] = ""
    save_bucket = "interrec"
    
    if "HTTP_PROXY" in os.environ:
        del os.environ["HTTP_PROXY"]
    if "HTTPS_PROXY" in os.environ:
        del os.environ["HTTPS_PROXY"]
    if "SOCKS_PROXY" in os.environ:
        del os.environ["SOCKS_PROXY"]
    # 设置代理
    proxy_url = "114.212.20.133:808"
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ["GRPC_DNS_RESOLVER"] = "native"
    os.environ["GRPC_PROXY"] = f"http://{proxy_url}"
    
    # 上传文件到 GCS
    file_name = input_file_path.split('/')[-1]
    print(f"正在上传文件 {input_file_path} 到 GCS {file_name}")
    upload_blob(save_bucket, input_file_path, file_name)
    
    # 构建 GCS 路径
    gcs_file_path = f"gs://{save_bucket}/{file_name}"
    print("文件路径: ", gcs_file_path)

    # 初始化 vertexai
    vertexai.init(project="gen-lang-client-0661527868", location="us-central1", api_transport="rest")
    output_uri = file_name.replace(".jsonl", "_output.jsonl")

    # 提交批量预测任务
    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-1.5-pro-002",
        input_dataset=gcs_file_path,
        output_uri_prefix=f"gs://{save_bucket}/{output_uri}",
    )

    # 打印任务信息
    print(f"任务资源名称: {batch_prediction_job.resource_name}")
    print(f"模型资源名称: {batch_prediction_job.model_name}")
    print(f"任务状态: {batch_prediction_job.state.name}")

    # 等待任务完成
    while not batch_prediction_job.has_ended:
        time.sleep(60)
        batch_prediction_job.refresh()
        print(f"刷新任务状态: {batch_prediction_job.state.name}")

    # 检查任务是否成功
    if batch_prediction_job.has_succeeded:
        print("任务成功！")
        print(f"输出位置: {batch_prediction_job.output_location}")
        download_file = batch_prediction_job.output_location.replace(f"gs://{save_bucket}/", "") + "/predictions.jsonl"
        
        output_path = input_file_path.replace(".jsonl", "_output.jsonl")
        print(f"正在下载文件 {download_file} 到 {output_path}")
        download_blob(save_bucket, download_file, output_path)
        return True
    else:
        print(f"任务失败: {batch_prediction_job.error}")
        return False

if __name__ == "__main__":
    input_file = "results/cds_raw_seedtest_sample_500_llama3_processed_multi_turn_batch_turn_33.jsonl"
    process_batch_prediction(input_file)