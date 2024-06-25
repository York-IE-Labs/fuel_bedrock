from ..common import BedrockInvocationParameters, ResponseBody
from json import dumps
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Optional


class TextGenerationModelConfig(BaseModel):
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    topP: Optional[Annotated[float, Field(strict=True, gt=0, le=1)]] = None
    maxTokenCount: Optional[Annotated[int, Field(strict=True, ge=0, le=8_000)]] = None
    stopSequences: Optional[List[str]] = None


class TextGenerationBody(BaseModel):
    inputText: str
    textGenerationConfig: Optional[TextGenerationModelConfig] = None


class TextGenerationConfig(BedrockInvocationParameters):
    modelId: Literal['amazon.titan-text-lite-v1', 'amazon.titan-text-express-v1']
    body: TextGenerationBody

    @property
    def body_payload(self):
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class TextGenerationResponse(ResponseBody):
    def get_result(self) -> str:
        return self.body.get("results", (dict(), ))[0].get("outputText", "")
