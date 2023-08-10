import os
from tools.utils import jload
from transformers import pipeline, AutoConfig, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader

import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_CACHE"]="/scratch/huggingface"
SUMMARIZATION_MODEL = "t5-base"
SUMMARIZATION_CONFIG = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_length": 256,
    "min_length": 128,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
    "prefix": "summarize: "
}
MODEL_MAX_LENGTH = 10240 # a random large number for tokenization limit


class MerkManualDB:

    def __init__(
            self, 
            db_path, 
            device="cuda:2",
            emb_model_name="sentence-transformers/all-MiniLM-L6-v2", 
            cache_folder=".scratch", 
            persist_directory='.scratch/chroma/',
            ):
        self.database = jload(f=db_path)
        self.data = JSONLoader(
            file_path=db_path,
            jq_schema='.[]',
            text_content=False,
            ).load()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=emb_model_name,
            cache_folder=cache_folder,
            )  
        self.vectorstore = Chroma.from_documents(
            self.data, 
            self.embeddings, 
            persist_directory=persist_directory
            )
        self.summarizer = self.load_summarizer(device=device)
    
    def search(self, query, k=3, summarize=True):
            
        context = ""
        results = self.vectorstore.max_marginal_relevance_search(query, k=k)
        for result in results:
            disease_candidate = list(self.database.keys())[result.metadata["seq_num"]-1]
            info = result.page_content
            if summarize:
                # possible failures to this?
                info = self.summarizer(info)[0]["summary_text"]

            context += f"{disease_candidate}: {info}\n"
        return context

    @staticmethod
    def load_summarizer(device):
        config = AutoConfig.from_pretrained(
            SUMMARIZATION_MODEL, 
            cache_dir=os.environ["TRANSFORMERS_CACHE"]
            )
        config.task_specific_params["summarization"] = SUMMARIZATION_CONFIG
        return pipeline(
            task="summarization",
            model=SUMMARIZATION_MODEL,
            tokenizer=AutoTokenizer.from_pretrained(
                SUMMARIZATION_MODEL,
                model_max_length=MODEL_MAX_LENGTH,
            ),
            config=config,
            device=device
            )
