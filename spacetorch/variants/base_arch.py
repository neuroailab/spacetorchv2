import torch
from abc import ABC, abstractmethod


class BaseArch(ABC):
    """A base class for constructing all variants"""
    def __init__(self):
        self.cfg = None
        self.eval_cfg = None
        self.model = None
        self.eval_model = None
        self.base_model = None
        self.training_procotol = None
        self.eval_protocol = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def set_cfg(self, args):
        pass

    @abstractmethod
    def set_eval_cfg(self, args):
        pass

    @abstractmethod
    def set_model(self, args):
        pass

    @abstractmethod
    def set_eval_model(self, args):
        pass

    @abstractmethod
    def set_training_protocol(self, args):
        pass

    @abstractmethod
    def start_training_protocol(self):
        pass

    @abstractmethod
    def set_eval_protocol(self, args):
        pass

    @abstractmethod
    def start_eval_protocol(self):
        pass

    def train_init(self, args):
        self.set_cfg(args)
        self.set_model(args)
        self.set_training_protocol(args)
        
    def eval_init(self, args):
        self.set_eval_cfg(args)
        self.set_eval_model(args)
        self.set_eval_protocol(args)
