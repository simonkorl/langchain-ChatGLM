from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.faiss import dependable_faiss_import
from typing import Any, Callable, List, Dict, Tuple, Optional
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
import numpy as np
import copy
import os
from configs.model_config import *
from transformers import BertTokenizer, BertModel
import torch


class MyFAISS(FAISS, VectorStore):
    def __init__(
            self,
            embedding_function: Callable,
            index: Any,
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            normalize_L2: bool = False,
    ):
        super().__init__(embedding_function=embedding_function,
                         index=index,
                         docstore=docstore,
                         index_to_docstore_id=index_to_docstore_id,
                         normalize_L2=normalize_L2)
        self.score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE
        self.chunk_conent = False
        self.load_model_for_filename_similarity()
 
 
    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        # TODO: 增加是否属于同一文档的判断
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists
    

    def delete_doc(self, source: str or List[str]):
        try:
            if isinstance(source, str):
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] == source]
                vs_path = os.path.join(os.path.split(os.path.split(source)[0])[0], "vector_store")
            else:
                ids = [k for k, v in self.docstore._dict.items() if v.metadata["source"] in source]
                vs_path = os.path.join(os.path.split(os.path.split(source[0])[0])[0], "vector_store")
            if len(ids) == 0:
                return f"docs delete fail"
            else:
                for id in ids:
                    index = list(self.index_to_docstore_id.keys())[list(self.index_to_docstore_id.values()).index(id)]
                    self.index_to_docstore_id.pop(index)
                    self.docstore._dict.pop(id)
                # TODO: 从 self.index 中删除对应id
                # self.index.reset()
                self.save_local(vs_path)
                return f"docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"


    def update_doc(self, source, new_docs):
        try:
            delete_len = self.delete_doc(source)
            ls = self.add_documents(new_docs)
            return f"docs update success"
        except Exception as e:
            print(e)
            return f"docs update fail"

    def list_docs(self):
        return list(set(v.metadata["source"] for v in self.docstore._dict.values()))
    

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 5, selected_headings: List[str] = [] # Added loaded_files parameter - Luming modified 20230614
    ) -> List[Document]:
        #print("SUCCESS", loaded_files) #Luming modified 20230614
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        #print("OUTPUT vector:", vector, np.shape(vector))
        #print("OUTPUT index:", self.index, type(self.index))
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k) #FAISS OUTPUT:[[351.794   366.41095 366.41095 366.41095 366.41095]] [[4096 2986 3479 4476 4762]]
        print("FAISS OUTPUT:", scores, indices)
        docs = []
        #id_set = set()
        id_list = list()
        cur_docs_len = 0
        store_len = len(self.index_to_docstore_id) #14967: 'a72a815c-d874-4b3d-a422-fc4f674a828b', 14968: '6da0d05f-a45d-4535-8fc1-3665e5ac481e', 14969: '63b3963b-5c21-4887-8236-c9dd89e53b6e'} 14970
        #print("OUTPUT index_to_docstore_id:", self.index_to_docstore_id, store_len)
        rearrange_id_list = False
        min_index_score = 9999
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            if i in self.index_to_docstore_id:
                _id = self.index_to_docstore_id[i]
            # 执行接下来的操作
            else:
                continue

            if selected_headings:
                if min_index_score > scores[0][j]:
                    min_index_score = scores[0][j]
                elif scores[0][j] - min_index_score > 100: # The following docs are not that related to the query - Luming modified 20230703
                    break
            #print("OUTOUT _id:", _id)
            doc = self.docstore.search(_id)
            print("OUTPUT doc, index, id & score:", doc, i, _id, scores[0][j])

            # --------------------------Original code----------------------------
            # if (not self.chunk_conent) or ("context_expand" in doc.metadata and not doc.metadata["context_expand"]):
                # 匹配出的文本如果不需要扩展上下文则执行如下代码
            #    print("No extension")
            #    if not isinstance(doc, Document):
            #        raise ValueError(f"Could not find document for id {_id}, got {doc}")
             #   doc.metadata["score"] = int(scores[0][j])
             #   docs.append(doc)
            #    continue
            #----------------------------------------------------------------------

            # Luming modified 20230703
            # Now can retrieve the entire block from one heading to another
            docs_len = len(doc.page_content)
            cur_docs_len += docs_len
            if doc.page_content in selected_headings:
                print("IN if doc.page_content in selected_headings")
                if cur_docs_len < self.chunk_size*k:
                    id_list.append(i)
                    break_flag = False
                    cur_index = i+1
                    len_docstore = len(self.index_to_docstore_id) 
                    prev_doc0 = None
                    while cur_index < len_docstore and not break_flag:                
                        if cur_index not in id_list:
                            _id0 = self.index_to_docstore_id[cur_index]
                            doc0 = self.docstore.search(_id0)
                            #if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != \
                            #        doc.metadata["source"]:
                            if prev_doc0 != None and prev_doc0.page_content not in selected_headings:#deal with continuous headings
                                if doc0.page_content in selected_headings or doc0.metadata["source"] != doc.metadata["source"]:
                                    break_flag = True
                                    break
                                elif doc0.metadata["source"] == doc.metadata["source"]:
                                    docs_len += len(doc0.page_content)
                                    cur_docs_len += docs_len
                                    id_list.append(cur_index)
                                    rearrange_id_list = True
                            else:
                                    docs_len += len(doc0.page_content)
                                    cur_docs_len += docs_len
                                    id_list.append(cur_index)
                        if break_flag:
                            break
                        cur_index+=1
                        prev_doc0 = doc0
            else:  #follow the original code
                # Change id_set to id_list
                # Use a temp list to deal with sorting of a sub list
                
                temp_id_list = []
                temp_id_list.append(i)
                for k in range(1, max(i, store_len - i)):
                    break_flag = False
                    if "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "forward":
                        expand_range = [i + k]
                    elif "context_expand_method" in doc.metadata and doc.metadata["context_expand_method"] == "backward":
                        expand_range = [i - k]
                    else:
                        expand_range = [i + k, i - k]
                    for l in expand_range:
                        if l not in temp_id_list and 0 <= l < len(self.index_to_docstore_id):
                            _id0 = self.index_to_docstore_id[l]
                            doc0 = self.docstore.search(_id0)
                            if docs_len + len(doc0.page_content) > self.chunk_size or doc0.metadata["source"] != \
                                    doc.metadata["source"]:
                                break_flag = True
                                break
                            elif doc0.metadata["source"] == doc.metadata["source"]:
                                docs_len += len(doc0.page_content)
                                cur_docs_len += docs_len
                                temp_id_list.append(l)
                                rearrange_id_list = True
                    if break_flag:
                        break
                temp_id_list = sorted(temp_id_list)
                id_list += temp_id_list
            #print("OUTPUT temp docs_len", cur_docs_len)
        #print("Expand range:", expand_range)
        #if (not self.chunk_conent) or (not rearrange_id_list):
        #    print("IN return docx")
         #   return docs
        #if len(id_set) == 0 and self.score_threshold > 0:
        #    print("IN return []")
        #    return []
        #print("OUTPUT seperate_list before:", id_set)
        #id_list = sorted(list(id_set))
        #id_list = list(id_set)
        if len(id_list) == 0:
            return docs, 0
        id_lists = self.seperate_list(id_list)
        #print("OUTPUT id_list:", id_list)
        #print("OUTPUT seperate_list after:", id_lists)
        for id_seq in id_lists:
            #print("OUTPUT id_seq:", id_seq)
            if len(id_seq) <= 1:
                continue
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    # doc = self.docstore.search(_id)
                    doc = copy.deepcopy(self.docstore.search(_id))
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append(doc)
        return docs, cur_docs_len

        
    # -------------------------------------------------------------------------------------------------------
    # Luming modified 20230625
    # The code below is all for GK project
    # -------------------------------------------------------------------------------------------------------
    def load_model_for_filename_similarity(self):
        # Load model from HuggingFace Hub 
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.TOKENIZER = BertTokenizer.from_pretrained(
        #'shibing624/text2vec-base-chinese',
        "GanymedeNil/text2vec-large-chinese",
        )
        self.MODEL = BertModel.from_pretrained(
        #'shibing624/text2vec-base-chinese'
        "GanymedeNil/text2vec-large-chinese",
        )
        self.MODEL.to(self.device)

    
    #doc_name_mode= True: Compare doc names, else doc headings
    def compare_similarity_query_doc(self, query, doc_content_list, top_k=5, doc_name_mode=True): #if doc_name_mode=True, doc_content_list contains doc names
        max_doc_contents = []
        if(doc_name_mode):
            clean_doc_names = clean_loaded_filenames(doc_content_list)
            doc_embeddings = self.compute_embeddings(clean_doc_names)
        else:
            doc_embeddings = self.compute_embeddings(doc_content_list)

        #  Find maximum result
        max_indexes, max_similarities = self.find_max_results(query, doc_embeddings, k=top_k)
        for i in max_indexes:
            max_doc_contents.append(doc_content_list[i])
        print(f"Best Match: '{max_doc_contents}', similarity: {max_similarities}")
        return max_doc_contents
    
    def compute_embeddings(self, sentences: list[str]) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.TOKENIZER(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: tensor.to(self.device) for key, tensor in encoded_input.items()}
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.MODEL(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def find_max_results(self, input_sentence: str, embeddings: torch.Tensor, k: int = 2) -> tuple[int, float]:
        max_indexes = []
        max_similarities = []
        new_vector = self.compute_embeddings([input_sentence])
        # print("New Sentence embeddings:")
        # print(new_vector)

        # Calculate cosine similarity
        similarities = calculate_similarity(new_vector, embeddings)
        similarities = similarities.tolist()
        #print("OUTPUT File Similarities:", similarities)
        # Find top 2 maximum results
        temp_max_similarity = -1
        for i in range(0, k):
            max_similarity = max(similarities)
            if(max_similarity > temp_max_similarity):
                temp_max_similarity = max_similarity
            if max_similarity > 0.5:
                if(temp_max_similarity - max_similarity > 0.15):
                    break
                max_similarities.append(max_similarity)
                max_index = similarities.index(max_similarity)
                max_indexes.append(max_index)
                del similarities[max_index]

            else:
                continue
        
        return max_indexes, max_similarities #max_index.to('cpu').numpy(), max_similarity

    def read_docx_headings(self, f):
        from docx import Document
        headings = []
        #for f in loaded_files:
        cur_dir = os.getcwd()+r'/'+f
        #print("OUTPUT FILE", cur_dir)
        obj = Document(cur_dir)
        for p in obj.paragraphs:
            #style_name = p.style.name
            #print(style_name, p.text, len(p.text), sep=': ')
            if(p.style.name.startswith('Heading')):
                headings.append(p.text)
        return headings
    
    # Luming modified 20230614
    def similarity_search_with_score(
        self, query: str, k: int = 4, match_docs: List[str]=[]
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """ 
        embedding = self.embedding_function(query)
        if(match_docs):
            selected_headings = []
            for f in match_docs:
                selected_headings = self.read_docx_headings(f)
            print("OUTPUT headings:", selected_headings)
            docs, len_context = self.similarity_search_with_score_by_vector(embedding, k, selected_headings=selected_headings)
        else:
            print("OUTPUT Default generate answer")
            docs, len_context = self.similarity_search_with_score_by_vector(embedding, k)
        return docs, len_context
    
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def calculate_similarity(new_vectors: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    similarities = torch.nn.functional.cosine_similarity(
        new_vectors.reshape((1, -1)), embeddings
    )
    # print(similarities)
    return similarities

def clean_loaded_filenames(loaded_files):
    clean_loaded_files = []
    for filename in loaded_files:
        clean_loaded_files.append(filename.split("/")[-1].split(".")[-2])
    return clean_loaded_files

