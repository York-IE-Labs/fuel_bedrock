from pathlib import Path
from pydantic import BaseModel
from yaml import safe_load


class _Config(BaseModel):
    aws_region_name: str


with (Path(__file__).parent / "bedrock_runtime.yml").open() as f_in:
    BedrockRuntimeConfig = _Config(**safe_load(f_in))
