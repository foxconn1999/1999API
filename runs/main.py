import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from runs.model import Request_Content, Model_Manager
from runs.load_model import load_model_and_assets
from runs.predict_one import predict_one_text
from runs.config import *
from runs.src.utils import get_device, get_tokenizer

# global parameter
device = get_device()
tokenizer = get_tokenizer(MODEL_NAME)

# init the model_manager
model_manager = Model_Manager(device=device, tokenizer=tokenizer)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, device, model_manager

    model_manager.load(route=DATA_ROOT)
    yield
    model_manager.clear()


def inference(route: str, content: Request_Content):
    global model_manager

    model, label_inputs, index_to_label, args = model_manager.load(route=route)

    return predict_one_text(
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

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict_one(content: Request_Content):
    # return value
    output = {}

    """
    level1
    """
    level1_result = inference(route=DATA_ROOT, content=content)
    output["level1"] = {item[0]: round(item[1], 4) for item in level1_result}

    """
    level2
    """
    level2_result = {}

    for agenda, _ in output["level1"].items():
        if os.path.exists(f"./runs/level2/{agenda}"):
            agenda_result = inference(route=f"./runs/level2/{agenda}", content=content)
            level2_result[agenda] = {item[0]: round(item[1], 4) for item in agenda_result}
        else:
            level2_result[agenda] = "None"

    output["level2"] = level2_result

    """
    level3
    """
    level3_result = {}

    for agenda in output["level2"].keys():
        if output["level2"][agenda] != "None":
            level3_result[agenda] = {}
            for task, _ in output["level2"][agenda].items():
                if os.path.exists(f"./runs/level3/{agenda}/{task}"):
                    task_result = inference(route=f"./runs/level3/{agenda}/{task}", content=content)
                    level3_result[agenda][task] = {item[0]: round(item[1], 4) for item in task_result}
                else:
                    level3_result[agenda][task] = "None"
        else:
            level3_result[agenda] = "None"

    output["level3"] = level3_result

    return output