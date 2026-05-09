from fastapi import FastAPI
from contextlib import asynccontextmanager

from runs.model import Request_Content
from runs.load_model import load_model_and_assets
from runs.predict_one import predict_one_text
from runs.config import *

model = None
tokenizer = None
label_inputs = None
index_to_label = None
args = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, label_inputs, index_to_label, args, device

    model, tokenizer, label_inputs, index_to_label, args, device = load_model_and_assets(
        ckpt_path=CKPT_PTAH,
        data_path=DATA_ROOT,
    )

    yield
    del model
    del tokenizer
    del label_inputs
    del index_to_label
    del args
    del device

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict_one(content: Request_Content):
    # get the predict of model
    print(MAX_LENGTH)
    result = predict_one_text(
        text=content.Text,
        model=model,
        tokenizer=tokenizer,
        label_inputs=label_inputs,
        index_to_label=index_to_label,
        device=device,
        max_length=MAX_LENGTH,
        rho=RHO,
        top_k=content.K,
    )

    result_of_normalization = {item[0]: round(item[1], 4) for item in result}
    return result_of_normalization