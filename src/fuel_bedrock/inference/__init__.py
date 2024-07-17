from .ai21 import invoke_jurassic
from .amazon import invoke_titan
from .anthropic import invoke_claude
from .cohere import invoke_command
from .meta import invoke_llama
from .mistral import invoke_mistral

from ..models.inference.ai21 import TextGenerationConfig as Ai21TextGenerationConfig
from ..models.inference.amazon import TextGenerationConfig as AmazonTextGenerationConfig
from ..models.inference.anthropic import MessagesConfig as AnthropicMessagesConfig
from ..models.inference.anthropic import TextCompletionsConfig as AnthropicTextCompletionsConfig
from ..models.inference.cohere import TextGenerationConfig as CohereTextGenerationConfig
from ..models.inference.common import BedrockInvocationParameters, ResponseBody
from ..models.inference.meta import TextGenerationConfig as MetaTextGenerationConfig
from ..models.inference.mistral import TextGenerationConfig as MistralTextGenerationConfig


def invoke_llm(config: BedrockInvocationParameters) -> ResponseBody:

    if not isinstance(config, BedrockInvocationParameters):
        raise TypeError("config must be a subclass of BedrockInvocationParameters")

    match config:
        case ai21 if isinstance(config, Ai21TextGenerationConfig):
            fct = invoke_jurassic
        case amazon if isinstance(config, AmazonTextGenerationConfig):
            fct = invoke_titan
        case anthropic if isinstance(config, (AnthropicMessagesConfig, AnthropicTextCompletionsConfig)):
            fct = invoke_claude
        case cohere if isinstance(config, CohereTextGenerationConfig):
            fct = invoke_command
        case meta if isinstance(config, MetaTextGenerationConfig):
            fct = invoke_llama
        case mistral if isinstance(config, MistralTextGenerationConfig):
            fct = invoke_mistral
        case _:
            raise NotImplementedError("No method implemented for invocation of the provided LLM config")

    return fct(config)


def generate_text(config: BedrockInvocationParameters) -> ResponseBody:
    response = invoke_llm(config)
    return response





