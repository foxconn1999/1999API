import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from runs.model import *
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

    """載入第一階跟局處室模型"""
    model_manager.load(route=LEVEL1_ROOT)
    model_manager.load(route=AGENDA_ROOT)
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
def predict(content: Request_Content):
    # return value
    output = {}

    """
    agenda
    """
    agenda_result = inference(route=AGENDA_ROOT, content=content)
    output["agenda"] = {item[0]: round(item[1], 4) for item in agenda_result}

    """
    level1
    """
    level1_result = inference(route=LEVEL1_ROOT, content=content)
    output["level1"] = {item[0]: round(item[1], 4) for item in level1_result}

    """
    level2
    """
    level2_result = {}

    for category, _ in output["level1"].items():
        if os.path.exists(f"./runs/level2/{category}"):
            category_result = inference(route=f"./runs/level2/{category}", content=content)
            level2_result[category] = {item[0]: round(item[1], 4) for item in category_result}
        else:
            level2_result[category] = "None"

    output["level2"] = level2_result

    """
    level3
    """
    level3_result = {}

    for category in output["level2"].keys():
        if output["level2"][category] != "None":
            level3_result[category] = {}
            for task, _ in output["level2"][category].items():
                if os.path.exists(f"./runs/level3/{category}/{task}"):
                    task_result = inference(route=f"./runs/level3/{category}/{task}", content=content)
                    level3_result[category][task] = {item[0]: round(item[1], 4) for item in task_result}
                else:
                    level3_result[category][task] = "None"
        else:
            level3_result[category] = "None"

    output["level3"] = level3_result

    return output

@app.post("/particular_predict")
def particular_predict(content: Request_Content_Partiuclar):
    output = {}
    category = content.Category
    output["level1"] = category

    """
    level2
    """
    level2_result = {}

    if os.path.exists(f"./runs/level2/{category}"):
        category_result = inference(route=f"./runs/level2/{category}", content=content)
        level2_result[category] = {item[0]: round(item[1], 4) for item in category_result}
    else:
        level2_result[category] = "None"

    output["level2"] = level2_result

    """
    level3
    """
    level3_result = {}

    if output["level2"][category] != "None":
        level3_result[category] = {}
        for task, _ in output["level2"][category].items():
            if os.path.exists(f"./runs/level3/{category}/{task}"):
                task_result = inference(route=f"./runs/level3/{category}/{task}", content=content)
                level3_result[category][task] = {item[0]: round(item[1], 4) for item in task_result}
            else:
                level3_result[category][task] = "None"
    else:
        level3_result[category] = "None"

    output["level3"] = level3_result

    return output