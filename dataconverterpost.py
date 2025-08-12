import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import argparse
import os.path as osp
import os
parser = argparse.ArgumentParser()
parser.add_argument("dstpath", type=str)
parser.add_argument("--ncpu", type=int, default=1)
parser.add_argument("--model-dim", type=int, default=512)
args = parser.parse_args()


class EdgeEmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            device="cuda:0",
            cache_folder="cache_data/model",
            trust_remote_code=True,
            truncate_dim=args.model_dim,
        )

    def encode(self, edgetype):
        # Only accept one string
        assert isinstance(edgetype, str)
        embedding = self.model.encode(
            edgetype,
            batch_size=1,
            convert_to_tensor=False,
            convert_to_numpy=True,
            prompt="clustering: ",
        )
        embedding = embedding.reshape(1, -1)
        embedding = embedding / np.linalg.norm(embedding, axis=1)[:, np.newaxis]
        embedding = embedding.reshape(-1)
        embedding = torch.from_numpy(embedding)
        return embedding

    def clean(self):
        del self.model
        torch.cuda.empty_cache()


relation_embedding_model = EdgeEmbeddingModel()

if os.path.exists(osp.join(args.dstpath, "edgenameemb.pt")):
    edgetypeemb = torch.load(osp.join(args.dstpath, "edgenameemb.pt"), map_location="cpu", weights_only=True)
else:
    edgetypeemb = {}

for key in edgetypeemb:
    edgetypeemb[key] = relation_embedding_model.encode(key)

edgetypeemb["fewshot"] = relation_embedding_model.encode("fewshot")

torch.save(edgetypeemb, osp.join(args.dstpath, "edgenameemb.pt"))