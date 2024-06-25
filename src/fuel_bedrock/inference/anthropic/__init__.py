from ...clients import bedrock_runtime_client, get_bedrock_results
from ...models.inference.anthropic import MessagesConfig, TextCompletionsConfig, MessagesResponse, TextCompletionsResponse
from ...models.inference.anthropic import MessagesBody, TextCompletionsBody
from ...models.inference.anthropic import TextContent, ImageContent
from json import loads
from typing import List


def _invoke_messages(config: MessagesConfig) -> MessagesResponse:
    response = bedrock_runtime_client.invoke_model(
        modelId=config.modelId,
        body=config.body_payload,
        accept=config.accept,
        contentType=config.contentType
    )
    response_body = get_bedrock_results(response)

    return MessagesResponse(body=response_body)


def _invoke_completions(config: TextCompletionsConfig) -> TextCompletionsResponse:
    response = bedrock_runtime_client.invoke_model(
        modelId=config.modelId,
        body=config.body_payload
    )

    response_body = loads(response.get('body').read())

    return TextCompletionsResponse(body=response_body)


def invoke_claude(config: MessagesConfig | TextCompletionsConfig) -> MessagesResponse | TextCompletionsResponse:
    assert isinstance(config, (MessagesConfig, TextCompletionsConfig))

    return (
        _invoke_completions(config)
        if isinstance(config, TextCompletionsConfig)
        else _invoke_messages(config)
    )


def create_message(role: str, content: List[dict]) -> dict:
    return {
        "role": role,
        "content": list(map(lambda d: TextContent(**d) if 'text' in d else ImageContent(**d), content))
    }


def get_completions_body(prompt: str, **kwargs) -> TextCompletionsBody:
    return TextCompletionsBody(prompt=prompt, **kwargs)


def get_messages_body(messages: List[dict], **kwargs) -> MessagesBody:
    messages = list(map(lambda m: create_message(m['role'], m['content']), messages))
    return MessagesBody(messages=messages, **kwargs)

