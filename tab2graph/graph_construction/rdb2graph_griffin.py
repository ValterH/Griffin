import pydantic

from .er_graph_construction_griffin import (
    ERGraphGriffinConstruction,
    ERGraphGriffinConstructionConfig
)

class RDB2GraphGriffinConfig(pydantic.BaseModel):
    # Whether to construct a relation table as edges.
    # If not, all tables will be constructed as nodes.
    relation_table_as_edge : bool = False
    # multi_enhance means whether we use multi_enhance module
    multi_enhance: bool = False
    # class_node_reverse_edge means whether we add reverse edge for class node to allow message-passing from data nodes to class nodes
    class_node_reverse_edge: bool = False
    # use_time_feature means whether we use time feature as sub_feature nodes
    use_time_feature: bool = False
    # add_fewshot_edges means whether we add few-shot edges for few-shot learning
    add_fewshot_edges: bool = False
    # keep_label_feature means we keep the label feature in seed feature if target node only has foreign key
    keep_label_feature: bool = False

class RDB2GraphGriffin(ERGraphGriffinConstruction):

    config_class = RDB2GraphGriffinConfig
    name = "r2n-griffin"
