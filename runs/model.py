from pydantic import BaseModel
from runs.load_model import load_model_and_assets

class Request_Content(BaseModel):
    Text: str
    K: int

class Request_Content_Partiuclar(Request_Content):
    Category: str

class Model_Manager():
    def __init__(self, tokenizer, device):
        self.cache = {}
        self.tokenizer = tokenizer
        self.device = device

    def load(self, route):
        if route not in self.cache:
            self.cache[route] = load_model_and_assets(
                ckpt_path=f"{route}/target/best.pt",
                data_path=route,
                tokenizer=self.tokenizer,
                device=self.device
            )

        return self.cache[route]
    
    def clear(self):
        self.cache.clear()