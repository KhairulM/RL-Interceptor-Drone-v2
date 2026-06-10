import abc
import torch
import torch.nn as nn
from torchrl.data import TensorSpec
from typing import Dict


class PlannerBase(nn.Module):

    REGISTRY: Dict[str, "PlannerBase"] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in PlannerBase.REGISTRY:
            raise ValueError("")
        super().__init_subclass__(**kwargs)
        PlannerBase.REGISTRY[cls.__name__] = cls
        PlannerBase.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def plan(self, *args, **kwargs) -> torch.Tensor:
        ...
