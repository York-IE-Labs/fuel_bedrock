from ..common import BedrockInvocationParameters, ResponseBody
from json import dumps
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Optional


class RepetitionPenalty(BaseModel):
    scale: float
    applyToWhitespaces: Optional[bool] = None
    applyToPunctuations: Optional[bool] = None
    applyToNumbers: Optional[bool] = None
    applyToStopwords: Optional[bool] = None
    applyToEmojis: Optional[bool] = None


class CountPenalty(RepetitionPenalty):
    scale: Annotated[float, Field(strict=True, ge=0, le=1)]


class PresencePenalty(RepetitionPenalty):
    scale: Annotated[float, Field(strict=True, ge=0, le=5)]


class FrequencyPenalty(RepetitionPenalty):
    scale: Annotated[float, Field(strict=True, ge=0, le=500)]


class TextGenerationBody(BaseModel):
    prompt: str
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    topP: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    maxTokens: Optional[Annotated[int, Field(strict=True, ge=0, le=8_191)]] = None
    stopSequences: Optional[List[str]] = None
    countPenalty: Optional[CountPenalty] = None
    presencePenalty: Optional[PresencePenalty] = None
    frequencyPenalty: Optional[FrequencyPenalty] = None


class TextGenerationConfig(BedrockInvocationParameters):
    modelId: Literal['ai21.j2-mid-v1', 'ai21.j2-ultra-v1']
    body: TextGenerationBody

    @property
    def body_payload(self):
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class TextGenerationResponse(ResponseBody):
    def get_result(self) -> str:
        return (
            self.body
            .get('completions', (dict(), ))[0]
            .get('data', dict())
            .get('text', "")
        )
