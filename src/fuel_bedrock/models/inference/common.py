from abc import ABC
from botocore.response import StreamingBody
from pydantic import BaseModel, Extra, Field


class BedrockInvocationParameters(BaseModel):
    modelId: str
    contentType: str = Field(default="application/json")
    accept: str = Field(default="application/json")
    body: str | dict


class ResponseBody(BaseModel):
    body: dict
    input_token_count: int
    output_token_count: int

    class Config:
        extra = Extra.allow

    def get_result(self) -> str:
        raise NotImplementedError
