# LatentCRS

## Datasets

The data processing code for three datasets is included in the `data` folder. Please download the datasets from the official websites:

- [Movielens-1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon CDs_and_Vinyl](https://amazon-reviews-2023.github.io/) and [Video_Games](https://amazon-reviews-2023.github.io/)

After downloading, place the files in the same folder as the code and run the preprocessing script. For the Amazon datasets, we also accept the 5-core setting and further filter items without descriptions in the metadata.

Or, you can download the processed data from [Baidu Disk](https://pan.baidu.com/s/1FcCbRxOCZRyp0sOTsaOtnQ?pwd=pp7f) or [Google Drive](https://drive.google.com/file/d/1FBH3DyI7SSCu_fAEZ7KV4Jc4eItdtyjY/view?usp=sharing).

## Requirements

- Python == 3.10
- PyTorch == 1.2.0
- tqdm == 4.26.0
- faiss-gpu == 1.7.1
- wandb
- hydra-core
- vertexai == 1.71.1

## Implementation

### Step 1: Preparation

The core of our approach combines large language models (LLMs) with traditional recommendation models based on the latent space of user intents. Therefore, the first step is to train a traditional recommendation model and extract the latent intent vectors.

We directly follow the official code of [ICLRec](https://github.com/salesforce/ICLRec) and save the cluster centers as intent vectors. All the preparation code can be found in the `pre_rec_model` folder. You can train the traditional recommendation model by running the following script:

```bash
bash pre_rec_model/run_<dataset_name>.sh
```

### Step 2: Caching for Efficient Training and Parameter Tuning

#### Get User Input via User Simulator

For more efficient training and parameter tuning, we cache user input from the user simulator during the first turn. The code to obtain responses from the user simulator is included in the `generated_data` folder, and the results are saved there as well.

For training, we randomly split the sequence of user behavior to augment the data. We fix the random seed to `1755, 2097, 3068, 6868, 7702` for reproducibility, but you can modify this in `data_generator/batch_data_simulator.py`.

For Gemini, we use the batch API from Google Cloud. You can refer to the [Official Guide](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#generative-ai-batch-text-python_vertex_ai_sdk) for more information. Before running the script, complete the following configuration in `data_generator/user_simulator_batch.sh`:

```bash
google_cloud_credentials
google_cloud_project
google_cloud_location
save_bucket
```

Once the configuration is complete, run the script to generate prompts for the user simulator and obtain responses. In this script, we randomly select 500 samples for multi-turn evaluation. Additionally, for the Amazon CD datasets, we randomly choose 1000 samples due to the high cost of evaluation.

```bash
bash data_generator/user_simulator_batch.sh
```

#### Cache Embeddings from LLM

Since we utilize LLM embeddings, we also cache these embeddings by running the following script:

```bash
bash text2embedding/cache_embedding.sh
```

### Step 3: EM Training

To perform Expectation-Maximization (EM) training, run the following script. This script also automatically evaluates the performance in the one-turn setting:

```bash
bash scripts/train_<dataset_name>.sh
```

### Multi-turn Setting Evaluation

All the code for multi-turn evaluation can be found in the `multi-turn` folder. First, fill in the dataset name and the saved model path in `multi-turn/multi_turn_evaluation.sh`, then run the script.

