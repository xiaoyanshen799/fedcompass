import abc
from typing import Dict, Union, OrderedDict, Tuple

class BaseAggregator:
    def increment_global_update(self):
        """Increase the number of completed global model updates."""
        if not hasattr(self, "_num_global_updates"):
            self._num_global_updates = 0
        self._num_global_updates += 1

    def get_num_global_updates(self) -> int:
        """Return the number of completed global model updates."""
        return getattr(self, "_num_global_updates", 0)

    def set_client_sample_size(self, client_id: Union[str, int], sample_size: int):
        """Set the sample size of a client"""
        if not hasattr(self, "client_sample_size"):
            self.client_sample_size = {}
        self.client_sample_size[client_id] = sample_size

    @abc.abstractmethod
    def aggregate(self, *args, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Aggregate local model(s) from clients and return the global model
        """
        pass

    @abc.abstractmethod
    def get_parameters(self, **kwargs) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return global model parameters"""
        pass
