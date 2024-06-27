from ..common import BedrockInvocationParameters, ResponseBody

from json import dumps
from pydantic import BaseModel, Field, PositiveInt, SerializeAsAny
from typing import Annotated, List, Literal, Optional


class MessageContent(BaseModel):
    pass


class ImageSource(BaseModel):
    type: str = Field(default="base64")
    media_type: str = Field(default="image/jpeg")
    data: str


class ImageContent(MessageContent):
    type: str = Field(default="image")
    source: ImageSource


class TextContent(MessageContent):
    type: str = Field(default="text")
    text: str


class Message(BaseModel):
    role: str
    content: List[MessageContent]


class MessagesBody(BaseModel):
    anthropic_version: str = Field(default="bedrock-2023-05-31")
    max_tokens: Optional[Annotated[int, Field(int, ge=0, le=200_000)]] = 500
    messages: List[Message]
    system: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_p: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_k: Optional[Annotated[int, Field(strict=True, ge=0, le=500)]] = None


class TextCompletionsBody(BaseModel):
    prompt: str
    max_tokens_to_sample: Optional[Annotated[int, Field(int, ge=0, le=200_000)]] = 500
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_p: Optional[Annotated[float, Field(strict=True, ge=0, le=1)]] = None
    top_k: Optional[Annotated[int, Field(strict=True, ge=0, le=500)]] = None


class TextCompletionsConfig(BedrockInvocationParameters):
    modelId: Literal['anthropic.claude-instant-v1', 'anthropic.claude-v2', 'anthropic.claude-v2:1']
    body: SerializeAsAny[TextCompletionsBody]

    @property
    def body_payload(self):
        return dumps(self.body.dict(exclude_none=True), separators=(",", ":"))


class MessagesConfig(BedrockInvocationParameters):
    modelId: Literal['anthropic.claude-instant-v1', 'anthropic.claude-v2', 'anthropic.claude-v2:1', 'anthropic.claude-3-haiku-20240307-v1:0', 'anthropic.claude-3-sonnet-20240229-v1:0']
    body: MessagesBody

    @property
    def body_payload(self):
        return bytes(dumps(self.body.dict(exclude_none=True), indent=0, separators=(",", ":")), 'utf-8')


class TextCompletionsResponse(ResponseBody):
    def get_result(self) -> str:
        return self.body.get("completion", "")


class MessagesResponse(ResponseBody):
    def get_result(self) -> str:
        return (
            self.body
            .get("content", (dict(), ))[0]
            .get("text", "")
        )
