import argparse
from omegaconf import OmegaConf
from appfl.agent import APPFLServerAgent
from appfl.comm.grpc import GRPCServerCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="config/server_fedavg.yaml",
    help="Path to the configuration file."
)
argparser.add_argument(
    "--logging-output-dir",
    type=str,
    default=None,
    help="Override server log output directory.",
)
argparser.add_argument(
    "--logging-output-filename",
    type=str,
    default=None,
    help="Override server log output base filename.",
)
args = argparser.parse_args()

server_agent_config = OmegaConf.load(args.config)
if args.logging_output_dir is not None:
    server_agent_config.server_configs.logging_output_dirname = args.logging_output_dir
if args.logging_output_filename is not None:
    server_agent_config.server_configs.logging_output_filename = args.logging_output_filename
server_agent = APPFLServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)
