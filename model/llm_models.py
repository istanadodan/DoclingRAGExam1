import os
from vertexai.generative_models import GenerativeModel
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


default_model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    # (you must create your safety settings using the previous step first).
    # "safety_settings": safety_settings,
}


def get_llm_model(model_name: str, temperature: float = 0.7):
    _kwargs = default_model_kwargs
    _kwargs["temperature"] = temperature

    return GenerativeModel(
        model_name=os.getenv(model_name),
        generation_config=_kwargs,
        system_instruction="You are a helpful assistant.",
    )


def get_embedding_model() -> Embeddings:
    """임베딩모델 선택

    Returns:
        Embeddings: 허깅페이스모델
    """
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBED_MODEL_ID"),
        model_kwargs=dict(trust_remote_code=True),
    )
