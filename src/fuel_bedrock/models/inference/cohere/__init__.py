from ..common import BedrockInvocationParameters, ResponseBody
from json import dumps
from pydantic import BaseModel, Field
from typing import Annotated, Dict, List, Literal, Optional


class TextGenerationBody(BaseModel):
    prompt: str
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=5)]] = None
    p: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    k: Optional[Annotated[float, Field(strict=True, ge=0, le=500)]] = None
    max_tokens: Optional[Annotated[int, Field(strict=True, ge=1, le=4_096)]] = None
    stop_sequences: Optional[List[str]] = None
    return_likelihoods: Optional[Literal['ALL', 'NONE', 'GENERATION']] = None
    stream: Optional[bool] = False
    num_generations: Optional[Annotated[int, Field(strict=True, ge=1, le=5)]] = None
    logit_bias: Optional[Dict[str | int, Annotated[float, Field(ge=-10, le=10)]]] = None
    truncate: Optional[Literal['NONE', 'START', 'END']]


class TextGenerationConfig(BedrockInvocationParameters):
    modelId: Literal['cohere.command-text-v14', 'cohere.command-light-text-v14']
    body: TextGenerationBody

    @property
    def body_payload(self):
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class TextGenerationResponse(ResponseBody):
    def get_result(self) -> str:
        return (
            self.body
            .get("generations", (dict(), ))[0]
            .get("text", "")
        )
