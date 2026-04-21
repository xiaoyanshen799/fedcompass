import argparse
from omegaconf import OmegaConf
from appfl.agent import APPFLClientAgent
from appfl.comm.grpc import GRPCClientCommunicator


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config",
    type=str,
    default="config/client_1.yaml",
    help="Path to the client configuration file.",
)
argparser.add_argument(
    "--client-id",
    type=int,
    default=None,
    help="Dataset partition id for this client.",
)
argparser.add_argument(
    "--num-clients",
    type=int,
    default=None,
    help="Override the total number of clients used for dataset partitioning.",
)
argparser.add_argument(
    "--logging-output-dir",
    type=str,
    default=None,
    help="Override client log output directory.",
)
argparser.add_argument(
    "--logging-output-filename",
    type=str,
    default=None,
    help="Override client log output base filename.",
)
argparser.add_argument(
    "--data-output-dir",
    type=str,
    default=None,
    help="Override dataset visualization output directory.",
)
argparser.add_argument(
    "--data-output-filename",
    type=str,
    default=None,
    help="Override dataset visualization output filename.",
)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(args.config)
if args.num_clients is not None:
    client_agent_config.data_configs.dataset_kwargs.num_clients = args.num_clients
if args.client_id is not None:
    client_agent_config.data_configs.dataset_kwargs.client_id = args.client_id
    client_agent_config.data_configs.dataset_kwargs.visualization = args.client_id == 0
    client_agent_config.train_configs.logging_id = f"Client{args.client_id + 1}"
if args.logging_output_dir is not None:
    client_agent_config.train_configs.logging_output_dirname = args.logging_output_dir
if args.logging_output_filename is not None:
    client_agent_config.train_configs.logging_output_filename = args.logging_output_filename
if args.data_output_dir is not None:
    client_agent_config.data_configs.dataset_kwargs.output_dirname = args.data_output_dir
if args.data_output_filename is not None:
    client_agent_config.data_configs.dataset_kwargs.output_filename = args.data_output_filename

client_agent = APPFLClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id=client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_communicator.invoke_custom_action(action="set_sample_size", sample_size=sample_size)

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    new_global_model, metadata = client_communicator.update_global_model(local_model)
    if metadata["status"] == "DONE":
        break
    if "local_steps" in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
    client_agent.load_parameters(new_global_model)
