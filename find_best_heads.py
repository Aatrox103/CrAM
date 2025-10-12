import os
import fire

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from utils.find_best_heads import *

if __name__ == "__main__":
    # calculating prob changes when changing different heads
    casual_tracing_per_head(
        filepath="nq_100_bge.json",
        LLM="YOUR_OWN_MODEL_PATH/Meta-Llama-3-8B-Instruct",
        output_dir="datasets/nq",
    )
    # calculating the mean prob change of each head across all data, and sorting the results accordingly
    casual_tracing_combine_all(input_path="datasets/nq/llama3")
    
    find_top_k_heads(input_path="datasets/nq/llama3", topk=100)
