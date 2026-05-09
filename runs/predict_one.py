import torch
import numpy as np

@torch.no_grad()
def predict_one_text(
    text,
    model,
    tokenizer,
    label_inputs,
    index_to_label,
    device,
    max_length=512,
    rho=0.05,
    top_k=10,
):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(
        doc_input_ids=input_ids,
        doc_attention_mask=attention_mask,
        label_input_ids=label_inputs["input_ids"].to(device),
        label_attention_mask=label_inputs["attention_mask"].to(device),
        rho=rho,
    )

    logits = out["logits"]
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    top_ids = np.argsort(-probs)[:top_k]
    top_labels = [(index_to_label[int(idx)], float(probs[int(idx)])) for idx in top_ids]

    return top_labels