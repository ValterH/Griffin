import yaml
import numpy as np
import os.path as osp
import torch
import copy
from torch_scatter import scatter_add
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("srcpath", type=str)
parser.add_argument("dstpath", type=str)
parser.add_argument("--ncpu", type=int, default=1)
args = parser.parse_args()

srcpath = args.srcpath #"griffin_datasets/joint-v52-pk-r2n/"#"tpberta-reg-r2n-combine/"#"./griffin_datasets/tpberta-bin-r2n-combine/"
dstpath = args.dstpath #"newdatasets/newds_v52_pk/"

with open(osp.join(dstpath, "metanode.yaml"), "r") as f:
    nodelist: dict = yaml.safe_load(f)

with open(osp.join(srcpath, "metadata.yaml"), "r") as f:
    config: dict = yaml.safe_load(f)
    
def edgename2tail(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[-1]
    elif edgename.startswith("tail of "):
        return edgename.split(":")[0].removeprefix("tail of ")
    else:
        raise NotImplementedError

def buildedgemeta(config, nodelist):
    metaadj = {nodetype: {"out": [], "in":[]} for nodetype in nodelist} # out, in
    edgelist = {nodetype: {} for nodetype in nodelist}
    for _ in config["graph"]["edges"]:
        path = _["path"]
        edgetype = _["type"]
        head, name, tail = edgetype.split(":")
        
        if name.endswith("-self_loop"):
            continue
        if name.startswith("reverse_"):
            continue
        
        metaadj[head]["out"].append(f"head of {edgetype}")
        metaadj[tail]["in"].append(f"tail of {edgetype}")
        try:
            assert f"head of {edgetype}" not in edgelist[head]
        except:
            print("! multi edge ", edgelist[head].keys(), head, tail, name)
        edgelist[head][f"head of {edgetype}"] = (0, path)

        if head!=tail:
            try:
                assert f"tail of {edgetype}" not in edgelist[tail]
            except:
                print("! multi edge ", edgelist[head].keys(), head, tail, name)
            edgelist[tail][f"tail of {edgetype}"] = (1, path)
    return metaadj, edgelist



metaadj, edgelist = buildedgemeta(config, nodelist)
with open(osp.join(dstpath, "metaadj.yaml"), "w") as f:
    yaml.dump(metaadj, f)


class EdgeEmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            device="cuda:0",
            cache_folder="cache_data/model",
            trust_remote_code=True,
            truncate_dim=512,
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

torch.save({
    edgetype: relation_embedding_model.encode(edgetype)
    for head in edgelist for edgetype in edgelist[head]
}, osp.join(dstpath, "edgenameemb.pt"))
relation_embedding_model.clean()
del relation_embedding_model

def adj2list(edge_index, num_node):
    idx = torch.argsort(edge_index[0])
    edge_index = edge_index[:, idx]
    src, tar = edge_index[0], edge_index[1]
    degree = scatter_add(torch.ones_like(src), src, dim=0, dim_size=num_node)
    ptr = src.new_zeros((num_node+1,))
    torch.cumsum(degree, dim=0, out=ptr[1:])
    return (ptr, tar)



from datasets import Dataset
from functools import partial

def getneighbor(x, edgedata, timestamps):
    id = x["number"]
    ret = {}
    for reltype in edgedata:
        ptr, tar = edgedata[reltype]
        nodelist = tar[ptr[id]:ptr[id+1]]
        ret[reltype] = nodelist
        # Get timestamp data and ensure consistent dtype
        timestamp_data = timestamps[reltype][nodelist]
        # Convert to float32 to avoid PyArrow mixing dtypes error
        if torch.is_tensor(timestamp_data):
            timestamp_data = timestamp_data.float()  # Convert to float32
        ret[reltype+"___TIMESTAMP"] = timestamp_data
    return ret

def process(nodetype):
    print(nodetype)
    edgelist[nodetype]
    if len(edgelist[nodetype]) == 0:
        return
    edgedata = {}
    num_node = nodelist[nodetype]["num"]
    for reltype in edgelist[nodetype]:
        pos, path = edgelist[nodetype][reltype]
        ei = torch.from_numpy(np.load(osp.join(srcpath, path)))
        if pos==1:
            ei = ei[[1, 0]]
        edgedata[reltype] = adj2list(ei, num_node)

    nodedss = {}
    for reltype in edgedata:
        timestamp_data = Dataset.load_from_disk(osp.join(dstpath, f"node/{edgename2tail(reltype)}/feat")).with_format("torch")["timestamp"]
        # Ensure consistent dtype to avoid PyArrow mixing dtypes error
        if torch.is_tensor(timestamp_data):
            timestamp_data = timestamp_data.float()  # Convert to float32
        nodedss[reltype] = timestamp_data
     
    ds = Dataset.from_dict({"number": list(range(num_node))})
    ds = ds.map(partial(getneighbor, edgedata=edgedata, timestamps=nodedss))
    ds.save_to_disk(osp.join(dstpath, "edge/", nodetype, "adj"))


from pqdm.processes import pqdm
#pqdm(list(edgelist.keys()), process, n_jobs=args.ncpu)
for nodetype in edgelist.keys():
    process(nodetype)

exit()