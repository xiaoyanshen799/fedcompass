from __future__ import annotations

import torchvision

from vision_dataset_utils import (
    class_noniid_partition,
    dirichlet_noniid_partition,
    get_torchvision_dataset,
    iid_partition,
    plot_distribution,
    plot_distribution_with_server,
)


def get_cifar10(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    include_server_distribution: bool = False,
    **kwargs,
):
    """
    Return the CIFAR-10 dataset for a given client.
    """
    return get_torchvision_dataset(
        torchvision.datasets.CIFAR10,
        num_clients=num_clients,
        client_id=client_id,
        partition_strategy=partition_strategy,
        include_server_distribution=include_server_distribution,
        **kwargs,
    )
