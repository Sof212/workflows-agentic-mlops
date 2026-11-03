from hashlib import sha256
import pickle
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os

class Preprocessing:
    def __init__(self,size_max=8000):
        self.cache_path = "/Users/sofianeagounikaci/Agentic-end-to-end-mlops/workflows-agentic-mlops/data/cache"
        self.types = ['.pdf', '.docx', '.txt']
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.size_max=size_max
    def _Validate_files(self, files):
        taille=sum(os.path.getsize(f) for f in files)
        if taille>self.size_max:
            raise ValueError(f"Les fichiers dépassent la taille maximale autorisée de {self.size_max} octets.")
        for f in files:
            if not any(f.endswith(ext) for ext in self.types):
                raise ValueError(f"Le fichier {f} n'est pas d'un type autorisé. Types autorisés : {self.types}")
        return True
    def get_all_chunks(self, files):
        self._Validate_files(files)
        chunks=[]
        chunks_hash=[]
        for f in files:
            with open(f, 'rb') as file:
                hash = sha256(file.read()).hexdigest()
            cache_file = os.path.join(self.cache_path, f"{hash}.pkl")
            if os.path.exists(cache_file):
                print("the file is already cached")
                chunks.append(self.load_from_cache(cache_file))


            else:
                chunk = self.get_chunks(f)
                self.save_to_cache(cache_file, chunk)
                chunks.append(chunk)
                chunks_hash.append(sha256(str(chunk).encode("utf-8")).hexdigest())

        return chunks

    def load_from_cache(self, cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data
    def save_to_cache(self, cache_file, chunks):
        with open(cache_file, "wb") as f:
            pickle.dump(chunks, f)
    def get_chunks(self, file:str) :
        """Original processing logic with Docling"""
        if  file.endswith(('.pdf', '.docx', '.txt', '.md')):

            converter = DocumentConverter()
            markdown = converter.convert(file).document.export_to_markdown()
            splitter = MarkdownHeaderTextSplitter(self.headers)
            return splitter.split_text(markdown)