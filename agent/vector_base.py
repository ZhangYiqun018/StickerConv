from datasets import load_dataset
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.retrievers import MultiQueryRetriever
from langchain.vectorstores import FAISS
import os
from typing import List, Dict
import random
import tenacity

class MemeVector:
    def __init__(self, db_path: str, model_name: str, llm: BaseLanguageModel, data_path: str=None, verbose: bool = True, force_create: bool = False, use_multi_query: bool = False) -> None:
        self.db_path = db_path
        self.embeddings = self.load_embeddings(model_name)
        if verbose:
            print(f"Load Embedding Model: {model_name} successful!")

        self.verbose = verbose
        if os.path.exists(db_path) and force_create is False:
            # 加载本地数据库
            if verbose:
                print(f"Dectect the DB: {db_path} is exists, Load the Local DB!")
            self.load_local_db()
        else:
            # 创建数据库
            if force_create and verbose:
                print(f"Force Create is True, Create a new db to {db_path}")
            if verbose and force_create is False:
                print(f"Do not Dectect the Local DB, create a new db to {db_path}")
            
            metadatas, text = self.load_data(data_path)
            self.create_db(metadatas=metadatas, text=text)
        
        if use_multi_query:
            assert llm != None, "You set multiquery, llm must not be NONE!"
            self.retriever = MultiQueryRetriever.from_llm(
                retriever = self.db.as_retriever(),
                llm       = llm,
            )
            self.retriever.verbose = verbose
        else:
            self.retriever = self.db.as_retriever()
            
    def load_data(self, data_path):
        dataset = load_dataset(
            'json',
            data_files=data_path,
            split = 'train'
        )
        metadatas = []
        text = []
        for idx, data in enumerate(dataset):
            # metadatas -> List[Dict]
            metadatas.append(
                {
                    "seq_num"       : idx + 1,
                    "image"         : data['image'],
                    "description"   : data['description'],
                    "emotion"       : data['emotion'],
                    "recommendation": data['recommendation'],
                    "origin_anno"   : data['origin_anno']
                }
            )
            # text.append(f"emotion: {data['origin_anno']}, emotion describe: {data['emotion']}, sticker describe: {data['sticker']}")
            text.append(f"emotion: {data['origin_anno']}, emotion describe: {data['emotion']}")
            
        if self.verbose:
            print(f"Load Data Success, Total Data Number: {len(text)}.")
            
        return metadatas, text
    
    def load_embeddings(self, model_name):
        return HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {"device": "cpu"},
        )
    
    def create_db(self, metadatas: List[Dict], text: List[str]):
        self.db = FAISS.from_texts(
            texts     = text,
            embedding = self.embeddings,
            metadatas = metadatas,
            ids       = range(len(text))
        )
        self.db.save_local(self.db_path)
    
    def load_local_db(self):
        self.db = FAISS.load_local(
            folder_path = self.db_path,
            embeddings = self.embeddings
        )
    
    def _search(self, query: str, k: int = 3):
        result = self.db.similarity_search_with_score(query, k=k)
        return result
    
    def _search_postprocess(self, results, low_score: float, high_score: float):
        filter_list = []
        for result in results:
            document, score = result
            if score >= low_score and score <= high_score:
                filter_list.append((score, document))
                
        return filter_list
    
    @tenacity.retry(reraise=True, stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=10, max=120))
    def search_retriever(self, query: str):
        retriever_result = self.retriever.get_relevant_documents(query)
        if retriever_result is None:
            return None
        if len(retriever_result) == 0:
            return None

        result = random.choice(retriever_result).metadata

        return result

    def search(self, query: str, low_score: float, high_score: float, k: int = 10):
        results = self._search(query, k)
        
        filter_list = self._search_postprocess(results=results, low_score=low_score, high_score=high_score)
        
        if len(filter_list) > 0:
            score, document =  random.choice(filter_list)
            return score, document.page_content, document.metadata
        
        return None, None, None
        
if __name__ == '__main__':
    from tqdm.auto import trange
    for i in trange(1, 21):
        myDB = MemeVector(
            db_path         = f"../dataset/vectorstore/test/split_{i}",
            model_name      = "../bge-large-en-v1.5",
            llm             = None,
            data_path       = f"/datas/zyq/research/chat_meme/dataset/labeled/test/sticker_{i}.json",
            force_create    = False,
            use_multi_query = False,
        )
    # # test
    # myDB.retriever.search_kwargs = {"k": 10}
    # documents = myDB.retriever.get_relevant_documents(
    #     query = "sad sticker"
    # )
    # print(documents)