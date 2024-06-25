from ..config.bedrock_runtime import BedrockRuntimeConfig
from boto3 import client as c
from botocore.config import Config as BotoConfig
from json import loads

runtime_boto_config = BotoConfig(region_name=BedrockRuntimeConfig.aws_region_name)
runtime_client = c("bedrock-runtime", config=runtime_boto_config)


def get_results(response: dict) -> dict:
    return loads(response.get('body', '{}').read())
