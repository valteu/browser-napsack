from napsack.label.clients.client import VLMClient, CAPTION_SCHEMA, IMAGE_CAPTION_SCHEMA, DENSE_CAPTION_SCHEMA, DENSE_IMAGE_CAPTION_SCHEMA
from napsack.label.clients.litellm import LiteLLMClient

try:
    from napsack.label.clients.bigquery import BigQueryClient, BigQueryResponse
except ImportError:
    BigQueryClient = None
    BigQueryResponse = None

try:
    from napsack.label.clients.tinfoil import TinfoilClient
except ImportError:
    TinfoilClient = None

def create_client(client_type: str, **kwargs) -> VLMClient:
    if client_type in ('gemini', 'litellm', 'vllm'):
        return LiteLLMClient(**kwargs)
    elif client_type == 'bigquery':
        if not BigQueryClient:
            raise ValueError("BigQuery dependencies not installed")
        return BigQueryClient(**kwargs)
    elif client_type == 'tinfoil':
        if not TinfoilClient:
            raise ValueError("Tinfoil dependencies not installed")
        return TinfoilClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

__all__ = [
    "VLMClient",
    "LiteLLMClient",
    "BigQueryClient",
    "BigQueryResponse",
    "TinfoilClient",
    "CAPTION_SCHEMA",
    "IMAGE_CAPTION_SCHEMA",
    "DENSE_CAPTION_SCHEMA",
    "DENSE_IMAGE_CAPTION_SCHEMA",
    "create_client",
]
