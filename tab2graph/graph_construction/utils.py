from enum import Enum
import pydantic
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import yaml


class GraphConstructionChoice(str, Enum):
    r2ne = "r2ne"
    r2n = "r2n"
    r2n_griffin = "r2n-griffin"


def get_graph_construction_class(graph_construction_name):
    if graph_construction_name == "r2ne":
        from .er_graph_construction import ERGraphConstruction

        graph_construction_class = ERGraphConstruction
    elif graph_construction_name == "r2n":
        from .rdb2graph import RDB2Graph

        graph_construction_class = RDB2Graph
    elif graph_construction_name == "r2n-griffin":
        from .rdb2graph_griffin import RDB2GraphGriffin

        graph_construction_class = RDB2GraphGriffin
    else:
        raise ValueError("Unknown graph construction name:", graph_construction_name)
    return graph_construction_class


class GriffinLabelFeatureConfig(pydantic.BaseModel):
    # LLM_name: str = "ST"
    LLM_name: str = "nomic"
    LLM_batch_size: int = 4096
    LLM_root_dir: str = "cache_data/model"


class GriffinLabelFeature:
    config_class = GriffinLabelFeatureConfig

    def __init__(self, dataset_name, config=GriffinLabelFeatureConfig()):
        self.config = config
        self.LLM_name = config.LLM_name
        self.LLM_batch_size = config.LLM_batch_size
        self.LLM_root_dir = config.LLM_root_dir
        # read TEMPLATE from yaml file by pydantic
        TEMPLATE_PATH = os.path.join(
            "configs/construct-graph/griffin_feature_template.yaml"
        )
        with open(TEMPLATE_PATH, "r") as f:
            TEMPLATE = yaml.safe_load(f)
        self.template = TEMPLATE.get(dataset_name, {}).get("template", {})
        self.task_label_template = TEMPLATE.get(dataset_name, {}).get("task_label_template", {})
        if self.LLM_name == "ST":
            self.model = SentenceTransformer(
                "multi-qa-distilbert-cos-v1",
                cache_folder=self.LLM_root_dir,
            )
            self.encode_with_template = self._ST_encode
            self.LLM_dim = 768
        elif self.LLM_name == "nomic":
            self.LLM_dim = 768
            self.model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                cache_folder=self.LLM_root_dir,
                trust_remote_code=True,
                truncate_dim=self.LLM_dim,
            )
            self.encode_with_template = self._nomic_encode

    def _query_template(self, text, task_name, target_column, category_mapping=None):
        if task_name in self.template:
            return self.template[task_name].format(
                self.task_label_template[task_name].get(text, "N/A")
            )
        else:
            text_template = f"The {target_column} is {{}}"
            return text_template.format(category_mapping.get(text, "N/A"))

    def encode_with_template(self, texts, task_name, category_mapping=None):
        raise NotImplementedError("Not define llm encoder yet")

    def _ST_encode(self, texts, task_name, target_column, category_mapping=None):
        texts = [self._query_template(text, task_name, target_column, category_mapping) for text in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.LLM_batch_size,
        ).astype("float32")
        return embeddings

    def _nomic_encode(self, texts, task_name, target_column, category_mapping=None):
        texts = [self._query_template(text, task_name, target_column, category_mapping) for text in texts]
        print(texts)
        embeddings = self.model.encode(
            texts,
            batch_size=self.LLM_batch_size,
            show_progress_bar=True,
            prompt="clustering: ",
        ).astype("float32")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        return embeddings


# TEMPLATE = {
#     "Joint-v3": {
#         "churn": "Task node of customers: the churn behavior is {}",
#         "kraj": "Task node of clients: located in the {} region.",
#         "charge": "Task node: advertisement expenditures charged from the wallet of service type {}",
#         "prepay": "Task node: advertisement expenditures prepaid into the wallet of service type {}",
#         "upvote": "Task node of posts: the upvote behavior is {}",
#         "amazon-churn": "Task node of customers: the churn behavior is {}",
#     },
#     "Joint-v4": {
#         "seznam-kraj": "Task node of clients: located in the {} region.",
#         "seznam-charge": "Task node: advertisement expenditures charged from the wallet of service type {}",
#         "seznam-prepay": "Task node: advertisement expenditures prepaid into the wallet of service type {}",
#         "stackexchange-churn": "Task node of users: the churn behavior is {}",
#         "stackexchange-upvote": "Task node of posts: the upvote behavior is {}",
#         "amazon-churn": "Task node of customers: the churn behavior is {}",
#         "retailrocket-cvr": "Task node of view: whether the item added to the cart is {}",
#         "retailrocket-available": "Task node of item availability: whether the item availability is {}",
#     },
#     "purchase": {
#         "discount": "Task node of products: the discount type is {}",
#         "attractiveness": "Task node of products: the attractiveness is {}",
#         "necessity": "Task node of products: the necessity is {}",
#         "rating_tendency": "Task node of customers: the rating tendency is {}",
#         "platform_engagement": "Task node of customers: the platform engagement is {}",
#         "spending_priority": "Task node of customers: the spending priority is {}",
#     },
#     "purchase-anonymous": {
#         "discount": "Task node of products: the discount type is {}",
#         "attractiveness": "Task node of products: the attractiveness is {}",
#         "necessity": "Task node of products: the necessity is {}",
#         "rating_tendency": "Task node of customers: the rating tendency is {}",
#         "platform_engagement": "Task node of customers: the platform engagement is {}",
#         "spending_priority": "Task node of customers: the spending priority is {}",
#     },
#     "facebook-recruiting": {
#         "bot": "Task node of bidders: whether or not the bidder is a robot is {}",
#     },
#     "redhat": {
#         "bval_pred": "Task node of action: the outcome of the action is {}",
#     },
#     "talkingdata": {
#         "demo-pred": "Task node of Gender_age: the user's the gender and age group is {}",
#     },
#     "elo-merchant-recommendation": {
#         "recommendation": "Task node of Card: the loyalty score of the card_id is {}",
#     },
#     "airbnb": {
#         "destination": "Task node of User: the destination country of the user is {}",
#     },
