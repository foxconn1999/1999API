from pydantic import BaseModel

class Request_Content(BaseModel):
    Text: str
    K: int