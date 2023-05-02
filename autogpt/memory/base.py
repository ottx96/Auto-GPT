"""Base class for memory providers."""
import abc

from gpt4free import forefront

from autogpt.config import AbstractSingleton, Config

cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return forefront.Embedding.create(prompt=[text], model="text-embedding-ada-002", token=cfg.forefront_api_key)[
        "data"
    ][0]["embedding"]


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
