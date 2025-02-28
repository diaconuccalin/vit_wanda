from typing import List, TypedDict

from the_models.Llama.Message import Message


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
