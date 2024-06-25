from ...clients import bedrock_runtime_client, get_bedrock_results
from ...models.inference.mistral import TextGenerationConfig, TextGenerationResponse


def invoke_mistral(config: TextGenerationConfig) -> TextGenerationResponse:

    assert isinstance(config, TextGenerationConfig)

    response = bedrock_runtime_client.invoke_model(
        modelId=config.modelId,
        body=config.body_payload,
        accept=config.accept,
        contentType=config.contentType
    )

    response_body = get_bedrock_results(response)

    return TextGenerationResponse(body=response_body)
