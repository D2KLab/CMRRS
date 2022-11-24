import numpy as np
from typing import List, Tuple
import faiss

class Indexer:
    def __init__(self, emb_size: int=512):
        # to get total length of flat index: index.xb.size()
        # to get number of embeddings in index: index.xb.size() // EMB_SIZE
        self.emb_size     = emb_size
        self.index        = faiss.index_factory(emb_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = 100
        self.faissId2MVId = []

    def add_content(self, content_embedding: np.ndarray, content_id: str):
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == self.emb_size, 'Expected embedding size of {}, got {}'.format(self.emb_size, content_embedding.shape[-1])
        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)
        self.index.add(content_embedding)
        self.faissId2MVId.append(content_id)

    def retrieve(self, query_embedding: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        query_embedding            = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        similarities, contents_idx = self.index.search(query_embedding, k)
        # assuming only one query
        similarities               = similarities[0]
        contents_idx               = contents_idx[0] #faiss internal indices
        #faiss results are already sorted from highest to lowest
        #sorted_idx               = np.argsort(similarity)[::-1]
        #mv_content_ids           = [self.faissId2MVId[idx] for idx in contents_idx[sorted_idx]]
        #sorted_similarities      = similarity[sorted_idx]
        mv_content_ids             = [self.faissId2MVId[idx] for idx in contents_idx]
        
        return mv_content_ids, similarities