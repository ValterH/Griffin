import os.path as osp
from typing import Iterable, Union, Literal

import yaml
import torch
import torch.nn.functional as F
import datasets as hds

INF = 100000000
MAXINT64 = 1<<62
TIMESTAMPADJNAME = "___TIMESTAMP"

def normalize_emb(embeddings):
    return F.normalize(embeddings, p=2, dim=-1)

class Node:
    meta: dict
    feat: hds.Dataset
    textemb: hds.Dataset
    adj: Union[hds.Dataset, None]

    def __init__(self, meta, feat, textemb, adj, feat_dim=512) -> None:
        self.meta = meta
        self.feat = feat
        self.textemb = textemb
        self.adj = adj
        self.feat_dim = feat_dim
        assert len(self.feat) == self.meta["num"]
        if self.adj is not None:
            assert (
                len(self.adj) == self.meta["num"]
            ), f"adj has {len(self.adj)} while there are {self.meta['num']} nodes"

    def __len__(self):
        return self.meta["num"]

    @property
    def is_target(self):
        return self.meta["is_target"]

    @property
    def featlist(self):
        return self.meta["feat"]
    

    def getfeat(self, idx: Union[int, Iterable[int]], floatemb) -> torch.Tensor:
        data: dict = self.feat[idx]
        def unique_query_textemb(idx, input_dim=None):
            unique_idx, inv = torch.unique(idx, return_inverse=True)
            text_embeddings = self.textemb[unique_idx]["emb"][inv]
            if input_dim is not None:
                assert text_embeddings.shape[1] >= input_dim, f"input_dim {input_dim} is larger than text embedding dimension {text_embeddings.shape[1]}"
                text_embeddings = text_embeddings[:, :input_dim]
            return normalize_emb(text_embeddings, p=2, dim=-1)

        def unique_float_emb(val, input_dim=None):
            unique_val, inv = torch.unique(val, return_inverse=True)
            float_embeddings = floatemb(unique_val)[inv]
            if input_dim is not None:
                assert float_embeddings.shape[1] >= input_dim, f"input_dim {input_dim} is larger than float embedding dimension {float_embeddings.shape[1]}"
                float_embeddings = float_embeddings[:, :input_dim]
            return float_embeddings
        
        data = torch.stack(
            [
                (
                    unique_query_textemb(data[_], input_dim=self.feat_dim)
                    if "Griffin_text_" in _
                    else unique_float_emb(data[_])#floatemb(data[_])
                )
                for _ in self.featlist
            ],
            dim=1,
        )  # (B, num_col, dim)
        return data

    def getedge(self, idx: torch.LongTensor, fanout: int = INF, timestamp: list[int] = None):
        if self.adj is None or self.is_target:
            return {}
        num_q = len(idx)
        # uni_idx, inv = torch.unique(idx, return_inverse=True)
        subadj: dict = self.adj[idx]# [uni_idx]
        subadj.pop("number")
        
        hastimestamp = timestamp is not None
        if hastimestamp:
            assert len(timestamp) == num_q
        ret = {}
        for key in subadj:
            if key.endswith(TIMESTAMPADJNAME):
                continue
            adj = subadj[key]
            if len(adj) == 0:
                continue
            if not isinstance(adj, torch.Tensor):
                assert isinstance(adj, list)
                adj = torch.nn.utils.rnn.pad_sequence(adj, batch_first=True, padding_value=-1)
            # adj = adj[inv]
            if len(adj.flatten()) == 0:
                continue
            assert adj.ndim == 2, f"{adj.shape}"
            assert adj.shape[0] == num_q
            rootnode = torch.arange(num_q).reshape(-1, 1).repeat(1, adj.shape[1])
            if hastimestamp:# and key + TIMESTAMPADJNAME in subadj:
                adjtimestamp = subadj[key + TIMESTAMPADJNAME]
                if not isinstance(adjtimestamp, torch.Tensor):
                    adjtimestamp = torch.nn.utils.rnn.pad_sequence(adjtimestamp, batch_first=True, padding_value=-1)
                # adjtimestamp = adjtimestamp[inv]
                assert adjtimestamp.ndim == 2
                assert adjtimestamp.shape[0] == num_q
                adj.masked_fill_(adjtimestamp>=timestamp.reshape(-1, 1), -1)
            
            if adj.shape[1] > fanout:
                rank_val = torch.rand_like(adj, dtype=torch.float)
                rank_val.masked_fill_(adj<0, -1000.)
                topk_ind = torch.topk(rank_val, fanout, dim=-1)[1]
                adj = torch.gather(adj, 1, topk_ind)
                rootnode = rootnode[:, :fanout]
            
            mask = adj >= 0
            rootnode, adj = rootnode[mask], adj[mask]
            if len(rootnode) == 0:
                continue
            ret[key] = torch.stack(
                (rootnode, adj), dim=0
            )
        return ret


def edgename2tail(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[-1]
    elif edgename.startswith("tail of "):
        return edgename.split(":")[0].removeprefix("tail of ")
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


def edgename2head(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[0].removeprefix("head of ")
    elif edgename.startswith("tail of "):
        return edgename.split(":")[-1]
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


class Graph:

    def __init__(self, path, feat_dim=512) -> None:
        with open(osp.join(path, "metanode.yaml")) as f:
            metanode = yaml.safe_load(f)
        with open(osp.join(path, "metaadj.yaml")) as f:
            metaadj = yaml.safe_load(f)
        for nodetype in metanode:
            metanode[nodetype].update(metaadj[nodetype])
        self.feat_dim = feat_dim
        self.metanode = metanode
        self.edgenameemb = torch.load(
            osp.join(path, "edgenameemb.pt"), map_location="cpu", weights_only=True
        )
        self.edgenameemb = {name: normalize_emb(emb[:feat_dim]) for name, emb in self.edgenameemb.items()}
        self.featnameemb = torch.load(
            osp.join(path, "featnameemb.pt"), map_location="cpu", weights_only=True
        )
        self.featnameemb = {name: normalize_emb(emb[:feat_dim]) for name, emb in self.featnameemb.items()}
        self.nodes = {
            nodetype: Node(
                self.metanode[nodetype],
                hds.load_from_disk(osp.join(path, "node", nodetype, "feat")).with_format("torch"),
                (
                    hds.load_from_disk(
                        osp.join(path, "node", nodetype, "textemb")
                    ).with_format("torch")
                    if osp.exists(osp.join(path, "node", nodetype, "textemb"))
                    else None
                ),
                (
                    None
                    if len(
                        self.metanode[nodetype]["in"] + self.metanode[nodetype]["out"]
                    )
                    == 0
                    else hds.load_from_disk(
                        osp.join(path, "edge", nodetype, "adj")
                    ).with_format("torch")
                ),
                feat_dim=self.feat_dim,
            )
            for nodetype in self.metanode
        }

    def subgraph(
        self,
        root_nodetype: str,
        root_nodeidx: Iterable[int],
        hop: int,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
    ):
        hastimestamp: bool = timestamp is not None
        adj = {}
        node = {}
        root = {root_nodetype: root_nodeidx}
        roottimestamp = {root_nodetype: timestamp}

        def dictgetlen(d: dict[str, torch.Tensor], key: str, dim: int = 0):
            if key not in d:
                return 0
            return d[key].shape[dim]

        def dictupdate(
            d: dict[str, torch.Tensor],
            key: str,
            value: torch.Tensor,
            concatdim: int = 0,
        ):
            if key not in d:
                d[key] = value
            else:
                d[key] = torch.concat((d[key], value), dim=concatdim)
            return d

        for h in range(hop):
            nroot, nroottimestamp = {}, {}
            for nodetype in root:
                found_src_num = dictgetlen(node, nodetype)
                # print(list(self.nodes.keys()), list(root.keys()), list(roottimestamp.keys()))
                ttadj = self.nodes[nodetype].getedge(
                    root[nodetype], fanout, roottimestamp[nodetype]
                )
                for edgetype in ttadj:
                    srcidx, taridx = ttadj[edgetype][0], ttadj[edgetype][1]

                    tartype = edgename2tail(edgetype)
                    found_tar_num = (
                        dictgetlen(node, tartype)
                        + dictgetlen(root, tartype)
                        + dictgetlen(nroot, tartype)
                    )

                    if hastimestamp:
                        tartimestamp = roottimestamp[nodetype][srcidx]
                        nroottimestamp = dictupdate(
                            nroottimestamp, tartype, tartimestamp
                        )
                    else:
                        nroottimestamp[tartype] = None
                    nroot = dictupdate(nroot, tartype, taridx)

                    srcidx = srcidx + found_src_num
                    taridx = (
                        torch.arange(taridx.shape[0], device=taridx.device)
                        + found_tar_num
                    )

                    adj = dictupdate(
                        adj, edgetype, torch.stack((srcidx, taridx), dim=0), concatdim=1
                    )

            for nodetype in root:
                node = dictupdate(node, nodetype, root[nodetype])
            del root
            del roottimestamp
            root = nroot
            roottimestamp = nroottimestamp

        for nodetype in root:
            node = dictupdate(node, nodetype, root[nodetype])

        mapping = torch.arange(len(root_nodeidx), device=root_nodeidx.device)
        # not change, latter code depends on mapping == arange

        for nodetype in node:
            assert len(node[nodetype]), f"{list(node.keys())} {root_nodeidx.shape} {list(adj.keys())}"
            #unique_idx, inv = torch.unique(node[nodetype], return_inverse=True)
            #node[nodetype] = self.nodes[nodetype].getfeat(unique_idx, floatemb)[inv]
            node[nodetype] = self.nodes[nodetype].getfeat(node[nodetype], floatemb)

        edgenameemb = {edgetype: self.edgenameemb[edgetype] for edgetype in adj}
        nodenameemb = {
            nodetype: torch.stack(
                [self.featnameemb[_] for _ in self.nodes[nodetype].featlist], dim=0
            )
            for nodetype in node
        }
        # print("loader: ", len(node), len(adj), list(nodenameemb.keys()), list(edgenameemb.keys()))

        return node, adj, nodenameemb, edgenameemb, mapping
    
    def fewshot(
        self,
        root_nodetype: str,
        root_nodeidx: torch.LongTensor,
        task_mask: torch.BoolTensor,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
        prefetch_factor: int = 10,
    ):
        assert fanout < INF, "alway sample past"
        # assert node idx in data is sorted by time stamp
        hastimestamp: bool = timestamp is not None

        idx = torch.randint(0, MAXINT64, (root_nodeidx.shape[0], prefetch_factor*fanout))
        idx = idx % (root_nodeidx.clamp_min(1)).reshape(-1, 1)
        
        if prefetch_factor > 1:
            rootfeat = self.nodes[root_nodetype].getfeat(root_nodeidx, floatemb) # (B, C, D)
            rootfeat[task_mask.unsqueeze(-1).expand(-1, -1, rootfeat.shape[-1])] = 0.
            feat = self.nodes[root_nodetype].getfeat(idx.flatten(), floatemb).unflatten(0, (-1, prefetch_factor*fanout)) # （B, FANOUT*PREFETCH, C, D)
            score = feat.flatten(-2, -1) @ rootfeat.flatten(-2, -1).unsqueeze(-1) # (B, FANOUT*PREFETCH, 1)
            score = score.squeeze(-1) # (B, FANOUT*PREFETCH)
            idx = torch.gather(idx, 1, torch.topk(score, k=fanout, dim=-1)[1]).flatten() # (B, FANOUT)
        idx = idx.flatten()
        rootnode = torch.arange(root_nodeidx.shape[0]).repeat_interleave(fanout)
        mask = root_nodeidx[rootnode] > 0
        idx, rootnode = idx[mask], rootnode[mask]
        return idx, rootnode


class Task:
    def __init__(self, path, feat_dim=512) -> None:
        with open(osp.join(path, "metatask.yaml")) as f:
            self.metatask = yaml.safe_load(f)

        self.feat_dim = feat_dim
        self.tasknameemb = torch.load(
            osp.join(path, "tasknameemb.pt"), map_location="cpu", weights_only=True
        )
        self.tasknameemb = {name: emb[:feat_dim] for name, emb in self.tasknameemb.items()}
        self.tasks = {
            taskname: hds.load_from_disk(osp.join(path, "task", taskname)).with_format(
                "torch"
            )
            for taskname in self.metatask
        }

    def get_retrieval(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "retrieval"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (meta["seed_type"], meta["num_class"]),
        )

    def get_regression(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "regression"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (None, None),
        )
