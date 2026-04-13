import io
import time
import numpy as np
import torch
import threading
import torch.nn as nn
from appfl.scheduler import *
from appfl.aggregator import *
from appfl.compressor import Compressor
from appfl.config import ServerAgentConfig
from appfl.misc import create_instance_from_file, get_function_from_file, run_function_from_file, set_random_seed
from appfl.logger import ServerAgentFileLogger
from concurrent.futures import Future
from omegaconf import OmegaConf, DictConfig
from typing import Union, Dict, OrderedDict, Tuple, Optional
from torch.utils.data import DataLoader

class APPFLServerAgent:
    """
    `APPFLServerAgent` should act on behalf of the FL server to:
    - provide configurations that are shared among all clients to the clients (e.g. trainer, model, etc.) `APPFLServerAgent.get_client_configs`
    - take the local model from a client, update the global model, and return it `APPFLServerAgent.global_update`
    - provide the global model to the clients (no input and no aggregation) `APPFLServerAgent.get_parameters`

    User can overwrite any class method to customize the behavior of the client agent.
    """
    def __init__(
        self,
        server_agent_config: ServerAgentConfig = ServerAgentConfig()
    ) -> None:
        self.server_agent_config = server_agent_config
        self._server_start_time: Optional[float] = None
        self._registered_clients: set[Union[int, str]] = set()
        self._registered_clients_lock = threading.Lock()
        if hasattr(self.server_agent_config.client_configs, "comm_configs"):
            self.server_agent_config.server_configs.comm_configs = (OmegaConf.merge(
                self.server_agent_config.server_configs.comm_configs,
                self.server_agent_config.client_configs.comm_configs
            ) if hasattr(self.server_agent_config.server_configs, "comm_configs") 
            else self.server_agent_config.client_configs.comm_configs
            )
        self._apply_random_seed()
        self._create_logger()
        self._load_model()
        self._load_loss()
        self._load_metric()
        self._load_server_validation()
        self._load_scheduler()
        self._load_compressor()

    def get_client_configs(self, **kwargs) -> DictConfig:
        """Return the FL configurations that are shared among all clients."""
        return self.server_agent_config.client_configs
    
    def global_update(
        self, 
        client_id: Union[int, str],
        local_model: Union[Dict, OrderedDict, bytes],
        blocking: bool = False,
        **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Update the global model using the local model from a client and return the updated global model.
        :param: client_id: A unique client id for server to distinguish clients, which be obtained via `ClientAgent.get_id()`.
        :param: local_model: The local model from a client, can be serailzed bytes.
        :param: blocking: The global model may not be immediately available for certain aggregation methods (e.g. any synchronous method).
            Setting `blocking` to `True` will block the client until the global model is available. 
            Otherwise, the method may return a `Future` object if the most up-to-date global model is not yet available.
        """
        if self.training_finished(internal_check=True):
            global_model = self.scheduler.get_parameters(init_model=False)
            return global_model
        else:
            if isinstance(local_model, bytes):
                local_model = self._bytes_to_model(local_model)
            global_model = self.scheduler.schedule(client_id, local_model, **kwargs)
            if not isinstance(global_model, Future):
                self._maybe_run_server_validation()
                return global_model
            if blocking:
                result = global_model.result() # blocking until the `Future` is done
                self._maybe_run_server_validation()
                return result
            else:
                return global_model # return the `Future` object
        
    def get_parameters(
        self, 
        blocking: bool = False,
        **kwargs
    ) -> Union[Future, Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """Return the global model to the clients."""
        global_model = self.scheduler.get_parameters(**kwargs)
        if not isinstance(global_model, Future):
            return global_model
        if blocking:
            return global_model.result() # blocking until the `Future` is done
        else:
            return global_model # return the `Future` object
        
    def set_sample_size(
            self, 
            client_id: Union[int, str],
            sample_size: int,
            sync: bool = False,
            blocking: bool = False,
        ) -> Optional[Union[Dict, Future]]:
        """
        Set the size of the local dataset of a client.
        :param: client_id: A unique client id for server to distinguish clients, which can be obtained via `ClientAgent.get_id()`.
        :param: sample_size: The size of the local dataset of a client.
        :param: sync: Whether to synchronize the sample size among all clients. If `True`, the method can return the relative weight of the client.
        :param: blocking: Whether to block the client until the sample size of all clients is synchronized. 
            If `True`, the method will return the relative weight of the client.
            Otherwise, the method may return a `Future` object of the relative weight, which will be resolved 
            when the sample size of all clients is synchronized.
        """
        self._mark_client_ready(client_id)
        if sync:
            num_clients = self._get_num_clients()
        self.aggregator.set_client_sample_size(client_id, sample_size)
        if sync:
            if not hasattr(self, "_client_sample_size"):
                self._client_sample_size = {}
                self._client_sample_size_future = {}
                self._client_sample_size_lock = threading.Lock()
            with self._client_sample_size_lock:
                self._client_sample_size[client_id] = sample_size
                future = Future()
                self._client_sample_size_future[client_id] = future
                if len(self._client_sample_size) == num_clients:
                    total_sample_size = sum(self._client_sample_size.values())
                    for client_id in self._client_sample_size_future:
                        self._client_sample_size_future[client_id].set_result(
                            {"client_weight": self._client_sample_size[client_id] / total_sample_size}
                        )
                    self._client_sample_size = {}
                    self._client_sample_size_future = {}
            if blocking:
                return future.result()
            else:
                return future
        return None

    def training_finished(self, internal_check: bool = False) -> bool:
        """Notify the client whether the training is finished."""
        finished = self.server_agent_config.server_configs.num_global_epochs <= self.scheduler.get_num_global_epochs()
        if finished and not internal_check:
            if not hasattr(self, "num_finish_calls"):
                self.num_finish_calls = 0
                self._num_finish_calls_lock = threading.Lock()
            with self._num_finish_calls_lock:
                self.num_finish_calls += 1
        return finished
    
    def server_terminated(self):
        """Whether the server can be terminated from listening to the clients."""
        if not hasattr(self, "num_finish_calls"):
            return False
        num_clients = self._get_num_clients()
        with self._num_finish_calls_lock:
            terminated = self.num_finish_calls >= num_clients
        if terminated and hasattr(self.scheduler, "clean_up"):
            self.scheduler.clean_up()
        return terminated

    def _create_logger(self) -> None:
        kwargs = {}
        if hasattr(self.server_agent_config.server_configs, "logging_output_dirname"):
            kwargs["file_dir"] = self.server_agent_config.server_configs.logging_output_dirname
        if hasattr(self.server_agent_config.server_configs, "logging_output_filename"):
            kwargs["file_name"] = self.server_agent_config.server_configs.logging_output_filename
        self.logger = ServerAgentFileLogger(**kwargs)

    def _apply_random_seed(self) -> None:
        """Set a fixed random seed for reproducible server-side initialization when configured."""
        if hasattr(self.server_agent_config.server_configs, "seed"):
            set_random_seed(int(self.server_agent_config.server_configs.seed))

    def _load_model(self) -> None:
        """
        Load model from the definition file, and read the source code of the model for sendind to the client.
        User can overwrite this method to load the model from other sources.
        """
        model_configs = self.server_agent_config.client_configs.model_configs
        self.model = create_instance_from_file(
            model_configs.model_path,
            model_configs.model_name,
            **model_configs.model_kwargs
        )
        # load the model source file and delete model path
        with open(model_configs.model_path, 'r') as f:
            self.server_agent_config.client_configs.model_configs.model_source = f.read()
        del self.server_agent_config.client_configs.model_configs.model_path

    def _load_loss(self) -> None:
        """
        Load loss function from various sources.
        - `loss_fn_path` and `loss_fn_name`: load the loss function from a file.
        - `loss_fn`: load the loss function from `torch.nn` module.
        - Users can define their own way to load the loss function from other sources.
        """
        if hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn_path") and hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn_name"):
            kwargs = self.server_agent_config.client_configs.train_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.server_agent_config.client_configs.train_configs.loss_fn_path,
                self.server_agent_config.client_configs.train_configs.loss_fn_name,
                **kwargs
            )
            with open(self.server_agent_config.client_configs.train_configs.loss_fn_path, 'r') as f:
                self.server_agent_config.client_configs.train_configs.loss_fn_source = f.read()
            del self.server_agent_config.client_configs.train_configs.loss_fn_path
        elif hasattr(self.server_agent_config.client_configs.train_configs, "loss_fn"):
            kwargs = self.server_agent_config.client_configs.train_configs.get("loss_fn_kwargs", {})
            if hasattr(nn, self.server_agent_config.client_configs.train_configs.loss_fn):
                self.loss_fn = getattr(nn, self.server_agent_config.client_configs.train_configs.loss_fn)(**kwargs)
            else:
                self.loss_fn = None
        else:
            self.loss_fn = None

    def _load_metric(self) -> None:
        """
        Load metric function from a file.
        User can define their own way to load the metric function from other sources.
        """
        if hasattr(self.server_agent_config.client_configs.train_configs, "metric_path") and hasattr(self.server_agent_config.client_configs.train_configs, "metric_name"):
            self.metric = get_function_from_file(
                self.server_agent_config.client_configs.train_configs.metric_path,
                self.server_agent_config.client_configs.train_configs.metric_name,
            )
            with open(self.server_agent_config.client_configs.train_configs.metric_path, 'r') as f:
                self.server_agent_config.client_configs.train_configs.metric_source = f.read()
            del self.server_agent_config.client_configs.train_configs.metric_path
        else:
            self.metric = None

    def _load_server_validation(self) -> None:
        """
        Optionally load a validation dataset for evaluating the global model on the server.
        """
        self.server_validation_enabled = False
        self.server_val_dataloader = None
        self._last_logged_global_update = 0
        self._server_validation_lock = threading.Lock()

        if not self.server_agent_config.server_configs.get("server_validation", False):
            return
        data_configs = None
        if hasattr(self.server_agent_config.server_configs, "validation_data_configs"):
            data_configs = self.server_agent_config.server_configs.validation_data_configs
        elif hasattr(self.server_agent_config.server_configs, "data_configs"):
            data_configs = self.server_agent_config.server_configs.data_configs
        elif hasattr(self.server_agent_config.client_configs, "data_configs"):
            data_configs = self.server_agent_config.client_configs.data_configs

        if data_configs is None:
            self.logger.info("Server validation disabled: missing validation_data_configs.")
            return
        if self.loss_fn is None or self.metric is None:
            self.logger.info("Server validation disabled: missing loss function or metric.")
            return

        dataset_kwargs = OmegaConf.to_container(
            data_configs.get("dataset_kwargs", {}),
            resolve=True,
        )
        if "client_id" in dataset_kwargs:
            dataset_kwargs["client_id"] = 0

        datasets = run_function_from_file(
            data_configs.dataset_path,
            data_configs.dataset_name,
            **dataset_kwargs,
        )
        if not isinstance(datasets, tuple) or len(datasets) < 2 or datasets[1] is None:
            self.logger.info("Server validation disabled: dataset did not return a validation split.")
            return

        train_configs = self.server_agent_config.client_configs.train_configs
        self.server_val_dataloader = DataLoader(
            datasets[1],
            batch_size=train_configs.get("val_batch_size", 32),
            shuffle=train_configs.get("val_data_shuffle", False),
            num_workers=train_configs.get("num_workers", 0),
        )
        self.server_validation_enabled = True
        self.logger.log_title(["Global Update", "Elapsed Time", "Val Loss", "Val Accuracy"])

    def _evaluate_global_model(self) -> Tuple[float, float]:
        """
        Evaluate the current global model on the server validation set.
        """
        assert self.server_val_dataloader is not None, "Validation dataloader is not initialized."
        device = self.server_agent_config.server_configs.get("device", "cpu")
        self.model.to(device)
        self.model.eval()
        val_loss = 0.0
        target_pred = []
        target_true = []
        with torch.no_grad():
            for data, target in self.server_val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.server_val_dataloader)
        val_accuracy = float(self.metric(np.concatenate(target_true), np.concatenate(target_pred)))
        self.model.train()
        return val_loss, val_accuracy

    def _maybe_run_server_validation(self) -> None:
        """
        Run server-side validation exactly once for each completed global model update.
        """
        if not self.server_validation_enabled:
            return
        current_update = self.aggregator.get_num_global_updates()
        if current_update <= self._last_logged_global_update:
            return

        with self._server_validation_lock:
            current_update = self.aggregator.get_num_global_updates()
            if current_update <= self._last_logged_global_update:
                return
            if self._server_start_time is None:
                # Fallback for flows that do not register all clients up front.
                self._server_start_time = time.time()
            val_loss, val_accuracy = self._evaluate_global_model()
            elapsed_time = time.time() - self._server_start_time
            self.logger.log_content([current_update, elapsed_time, val_loss, val_accuracy])
            self._last_logged_global_update = current_update

    def _get_num_clients(self) -> int:
        assert (
            hasattr(self.server_agent_config.server_configs, "num_clients") or
            hasattr(self.server_agent_config.server_configs, "scheduler_kwargs") and
            hasattr(self.server_agent_config.server_configs.scheduler_kwargs, "num_clients") or
            hasattr(self.server_agent_config.server_configs, "aggregator_kwargs") and
            hasattr(self.server_agent_config.server_configs.aggregator_kwargs, "num_clients")
        ), "The number of clients should be set in the server configurations."
        return (
            self.server_agent_config.server_configs.num_clients if
            hasattr(self.server_agent_config.server_configs, "num_clients") else
            self.server_agent_config.server_configs.scheduler_kwargs.num_clients if
            hasattr(self.server_agent_config.server_configs, "scheduler_kwargs") and
            hasattr(self.server_agent_config.server_configs.scheduler_kwargs, "num_clients") else
            self.server_agent_config.server_configs.aggregator_kwargs.num_clients
        )

    def _mark_client_ready(self, client_id: Union[int, str]) -> None:
        """
        Mark a client as ready once it has completed the initial handshake needed
        to start local training. The server elapsed-time clock starts when all
        expected clients have reached this point.
        """
        with self._registered_clients_lock:
            self._registered_clients.add(client_id)
            if (
                self._server_start_time is None and
                len(self._registered_clients) >= self._get_num_clients()
            ):
                self._server_start_time = time.time()
                self.logger.info(
                    "All expected clients registered; starting elapsed-time clock."
                )

    def _load_scheduler(self) -> None:
        """Obtain the scheduler."""
        self.aggregator: BaseAggregator = eval(self.server_agent_config.server_configs.aggregator)(
            self.model,
            OmegaConf.create(self.server_agent_config.server_configs.aggregator_kwargs),
            self.logger,
        )
        self.scheduler: BaseScheduler = eval(self.server_agent_config.server_configs.scheduler)(
            OmegaConf.create(self.server_agent_config.server_configs.scheduler_kwargs),
            self.aggregator,
            self.logger,
        )

    def _load_compressor(self) -> None:
        """Obtain the compressor."""
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.server_agent_config.server_configs, "comm_configs"):
            return
        if not hasattr(self.server_agent_config.server_configs.comm_configs, "compressor_configs"):
            return
        if getattr(self.server_agent_config.server_configs.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = Compressor(
                self.server_agent_config.server_configs.comm_configs.compressor_configs
            )

    def _bytes_to_model(self, model_bytes: bytes) -> Union[Dict, OrderedDict]:
        """Deserialize the model from bytes."""
        if not self.enable_compression:
            return torch.load(io.BytesIO(model_bytes))
        else:
            return self.compressor.decompress_model(model_bytes, self.model)
