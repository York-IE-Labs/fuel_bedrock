from ..config.bedrock_runtime import BedrockRuntimeConfig
from boto3 import client as c
from botocore.config import Config as BotoConfig
from json import loads
from typing import Tuple

runtime_boto_config = BotoConfig(region_name=BedrockRuntimeConfig.aws_region_name)
runtime_client = c("bedrock-runtime", config=runtime_boto_config)


def get_token_counts_from_headers(r: dict, default: int = -1) -> Tuple[int, int]:
    """
    parses token counts from bedrock resposne headers
    :param r: bedrock response
    :param default: sentinel for missing token counts
    :return: input_token_count, output_token_count
    """

    headers = r.get('ResponseMetadata', dict()).get("HTTPHeaders", dict())
    input_token_count = int(headers.get("x-amzn-bedrock-input-token-count", default))
    output_token_count = int(headers.get("x-amzn-bedrock-output-token-count", default))

    return input_token_count, output_token_count



def get_results(response: dict) -> dict:
    return loads(response.get('body', '{}').read())
