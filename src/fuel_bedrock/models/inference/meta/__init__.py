from ..common import BedrockInvocationParameters, ResponseBody
from json import dumps
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional


class TextGenerationBody(BaseModel):
    prompt: str
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_p: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    max_gen_len: Optional[Annotated[int, Field(strict=True, ge=1, le=4_096)]] = None


class TextGenerationConfig(BedrockInvocationParameters):
    modelId: Literal['meta.llama2-13b-chat-v1', 'meta.llama2-70b-chat-v1']
    body: TextGenerationBody

    @property
    def body_payload(self) -> str:
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class TextGenerationResponse(ResponseBody):
    def get_result(self) -> str:
        return self.body.get("generation", "")
