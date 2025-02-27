from typing import List, TypedDict

from models.Llama.Message import Message


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
