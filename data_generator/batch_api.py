import time
import vertexai

from vertexai.batch_prediction import BatchPredictionJob
import os


from google.cloud import storage
import os

import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--input_file_path", type=str, default="")
    parser.add_argument("--save_bucket", type=str, default="")
    parser.add_argument("--google_cloud_credentials", type=str, default="")
    parser.add_argument("--google_cloud_project", type=str, default="")
    parser.add_argument("--google_cloud_location", type=str, default="")
    args = parser.parse_args()
    # Initialize vertexai
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_cloud_credentials
    os.environ["GCLOUD_PROJECT"] = args.google_cloud_project
    os.environ["GCLOUD_LOCATION"] = args.google_cloud_location
    
    input_file_path = args.input_file_path
    save_bucket = args.save_bucket
    dataset_name = args.dataset
    
    input_file_path = f"generated_data/{dataset_name}/prompt4generation/{input_file_path}"
    
    print(f"Uploaded file {input_file_path} to GCS {input_file_path.split('/')[-1]}")
    upload_blob(save_bucket, input_file_path, input_file_path.split("/")[-1])
    gcs_file_path = f"gs://{save_bucket}/{input_file_path.split('/')[-1]}"
    print("file path: ", gcs_file_path)

    vertexai.init(project=os.environ["GCLOUD_PROJECT"], location=os.environ["GCLOUD_LOCATION"])
    output_uri = input_file_path.split('/')[-1].replace(".jsonl", "_output.jsonl")

    # Submit a batch prediction job with Gemini model
    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-1.5-flash-002",
        input_dataset=gcs_file_path,
        output_uri_prefix=f"gs://{save_bucket}/{output_uri}",
    )

    # Check job status
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    print(f"Model resource name with the job: {batch_prediction_job.model_name}")
    print(f"Job state: {batch_prediction_job.state.name}")

    # Refresh the job until complete
    while not batch_prediction_job.has_ended:
        time.sleep(10)
        batch_prediction_job.refresh()
        print(f"Refreshed Job state: {batch_prediction_job.state.name}")
    
    
    input_file_path = f"generated_data/{dataset_name}/prompt4generation/{input_file_path}"
    output_file_path = f"generated_data/{dataset_name}/gemini_generated/{input_file_path.replace('.jsonl', '_output.jsonl')}"
    
    
    # Check if the job succeeds
    if batch_prediction_job.has_succeeded:
        print("Job succeeded!")
        print(f"Job output location: {batch_prediction_job.output_location}")
        download_file = batch_prediction_job.output_location.replace(f"gs://{save_bucket}/", "") + "/" + "predictions.jsonl"
    
        print(f"Downloading file {download_file} to {output_file_path}")
        download_blob(save_bucket, download_file, output_file_path)
    else:
        print(f"Job failed: {batch_prediction_job.error}")
    # download_file = "gs://interrec/test_file_output.jsonl/prediction-model-2025-01-11T07:19:32.410981Z".replace(f"gs://{save_bucket}/", "") + "/" + "predictions.jsonl"
    # download_blob(save_bucket, download_file, input_file_path.replace(".jsonl", "_output.jsonl"))