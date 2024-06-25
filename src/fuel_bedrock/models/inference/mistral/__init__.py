from ..common import BedrockInvocationParameters, ResponseBody
from json import dumps
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Optional


class TextGenerationBody(BaseModel):
    prompt: str
    max_tokens: Optional[Annotated[int, Field(strict=True, ge=0, le=8_192)]] = None
    stop: Optional[List[str]] = None
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_p: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_k: Optional[Annotated[int, Field(strict=True, ge=1, le=200)]] = None


class TextGenerationConfig(BedrockInvocationParameters):
    modelId: Literal['mistral.mistral-7b-instruct-v0:2', 'mistral.mistral-8x7b-instruct-v0:1', 'mistral.mixtral-8x7b-instruct-v0:1', 'mistral.mistral-large-2402-v1:0']
    body: TextGenerationBody

    @property
    def body_payload(self) -> str:
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class TextGenerationResponse(ResponseBody):
    def get_result(self) -> str:
        return (
            self.body
            .get("outputs", (dict(), ))[0]
            .get("text", "")
        )
