import torch
import os

from transformers import AutoTokenizer

from runs.src.data import build_label_description_inputs
from runs.src.label_graph import maybe_load_ppmi
from runs.src.model import MLTCMedoidCLModel
from runs.src.utils import read_json

def build_index_to_label(label_to_index, num_labels):
    idx2 = [""] * num_labels
    for name, idx in label_to_index.items():
        idx2[idx] = str(name)
    return idx2

def load_model_and_assets(
    ckpt_path,
    data_path="./runs/foxconn_file",
    device=None,
    tokenizer=None
):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    dataset = args["dataset"]
    model_name = args["model_name"]
    dropout = args["dropout"]
    dgcn_layers = args["dgcn_layers"]
    eta = args["eta"]
    label_desc_max_length = args["label_desc_max_length"]

    label_to_index_path = os.path.join(data_path, "label_to_index.json")
    label_desc_path = os.path.join(data_path, f"label_desc_cache_{dataset}.json")

    label_to_index = read_json(label_to_index_path)
    num_labels = len(label_to_index)
    index_to_label = build_index_to_label(label_to_index, num_labels)
    label_desc = read_json(label_desc_path)

    label_inputs = build_label_description_inputs(
        tokenizer=tokenizer,
        index_to_label=index_to_label,
        label_desc=label_desc,
        max_length=label_desc_max_length,
    )

    ppmi_path = os.path.join(data_path, "cache", f"ppmi_eta{eta}.npy")
    ok, ppmi_adj = maybe_load_ppmi(ppmi_path)
    
    model = MLTCMedoidCLModel(
        model_name=model_name,
        num_labels=num_labels,
        ppmi_adj=torch.tensor(ppmi_adj, dtype=torch.float32),
        dgcn_layers=dgcn_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, label_inputs, index_to_label, args