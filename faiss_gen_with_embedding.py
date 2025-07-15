import os
import fire
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
import openai
import json
from sentence_transformers import SentenceTransformer
import pandas as pd


def main(filepath: str = "data/source/psgs_w100.tsv", topk: int = 10):

    paras = []
    # 21015324it
    # with open(filepath, 'r', encoding='utf-8') as f:
    #     k = 0
    #     for line in tqdm(f, total=21015324):
    #         if k > 100:
    #             break
    #         para = line.split("\t")[1]
    #         paras.append(para)
    #         k += 1

    # with open(filepath, 'r', encoding='utf-8') as f:
    #     for line in tqdm(f, total=21015324):
    #         para = line.split("\t")[1]
    #         paras.append(para)

    df = pd.read_csv(filepath,sep='\t')
    para=df['text'].to_list()

    model_name = "bge-large-en-v1.5"
    embedding_model = HuggingFaceEmbeddings(model_name=f"./model/{model_name}", model_kwargs={'device': 'cuda'}, encode_kwargs={'normalize_embeddings': True})
    if os.path.exists(f"./faiss_store/para-{model_name}/index.faiss") == False:
        os.makedirs(f"./faiss_store/para-{model_name}/", exist_ok=True)
        batch_size = 10000
        nbatch = len(paras) // batch_size + 1
        embeddings = []
        for k in tqdm(range(nbatch)):
            start_idx = k * batch_size
            end_idx = min((k + 1) * batch_size, len(paras))
            para = paras[start_idx:end_idx]
            embedding = embedding_model.embed_documents(para)
            embeddings.extend(embedding)
            if (k + 1) % 100 == 0:
                with open(f"./faiss_store/para-{model_name}/embeddings.pkl", 'wb') as f:
                    pickle.dump(embeddings, f)
                    print(f"Save embeddings at k={k}")

        with open(f"./faiss_store/para-{model_name}/embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
            print(f"Save all the embeddings")

        # with open(f"faiss_store/para-contriever-msmarco/embeddings.pkl", 'rb') as f:
        #     embeddings = pickle.load(f)

        text_embeddings = list(zip(paras, embeddings))
        vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embedding_model)
        # vector_store = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=embedding_model, distance_strategy="MAX_INNER_PRODUCT", normalize_L2=True)
        # vector_store = FAISS.from_texts(paras, embeddings)
        vector_store.save_local(f"./faiss_store/para-{model_name}/")
    else:
        vector_store = FAISS.load_local(f"./faiss_store/para-{model_name}/", embeddings=embedding_model)

    # question = "where is the capital of China"
    # docs = vector_store.similarity_search(question, k=topk)
    # contexts = [doc.page_content for doc in docs]


if __name__ == "__main__":
    fire.Fire(main)
