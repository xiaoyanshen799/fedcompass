from __future__ import annotations

import os
import pathlib
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from appfl.misc.data import Dataset
from torch.utils import data


def plot_distribution(
    num_clients: int,
    classes_samples: List[int],
    sample_matrix: np.ndarray,
    output_dirname: Optional[str],
    output_filename: Optional[str],
    row_labels: Optional[List[str]] = None,
):
    """
    Visualize the data distribution among clients for different classes.
    :param num_clients: number of clients
    :param classes_samples: number of samples for each class
    :param sample_matrix: the number of samples for each class for each client with shape (num_classes, num_clients)
    :param file_name: the filename to save the plot
    """
    _, ax = plt.subplots(figsize=(20, num_clients / 2 + 3))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ]

    for i in range(len(classes_samples)):
        ax.barh(
            y=range(num_clients),
            width=sample_matrix[i],
            left=np.sum(sample_matrix[:i], axis=0) if i > 0 else 0,
            color=colors[i],
        )

    if row_labels is None:
        row_labels = [f"Client{i + 1}" for i in range(num_clients)]
    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels(row_labels)
    output_dirname = "output" if output_dirname is None else output_dirname
    output_filename = "data_distribution.pdf" if output_filename is None else output_filename
    output_filename = (
        f"{output_filename}.pdf"
        if not output_filename.endswith(".pdf")
        else output_filename
    )
    if not os.path.exists(output_dirname):
        pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)
    unique = 1
    unique_filename = output_filename
    filename_base, ext = os.path.splitext(output_filename)
    while pathlib.Path(os.path.join(output_dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    plt.savefig(os.path.join(output_dirname, unique_filename))
    plt.close()


def _get_class_distribution(dataset: data.Dataset) -> np.ndarray:
    """Return class counts for a dataset as a 1D numpy array."""
    labels = []
    for _, label in dataset:
        labels.append(int(label))
    if not labels:
        return np.zeros(0, dtype=np.int32)
    num_classes = max(labels) + 1
    return np.bincount(labels, minlength=num_classes).astype(np.int32)


def plot_distribution_with_server(
    train_datasets: List[data.Dataset],
    server_dataset: data.Dataset,
    output_dirname: Optional[str],
    output_filename: Optional[str],
):
    """Plot client data distributions together with the server validation split."""
    client_distributions = [_get_class_distribution(dataset) for dataset in train_datasets]
    server_distribution = _get_class_distribution(server_dataset)
    num_classes = max([len(server_distribution)] + [len(dist) for dist in client_distributions])
    sample_matrix = np.zeros((num_classes, len(train_datasets) + 1), dtype=np.int32)
    for idx, dist in enumerate(client_distributions):
        sample_matrix[: len(dist), idx] = dist
    sample_matrix[: len(server_distribution), len(train_datasets)] = server_distribution
    row_labels = [f"Client{i + 1}" for i in range(len(train_datasets))] + ["Server"]
    output_filename = (
        None if output_filename is None
        else f"{pathlib.Path(output_filename).stem}_with_server.pdf"
    )
    plot_distribution(
        len(train_datasets) + 1,
        server_distribution.tolist(),
        sample_matrix,
        output_dirname,
        output_filename,
        row_labels=row_labels,
    )


def iid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
) -> List[data.Dataset]:
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in an IID manner.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    train_dataset_split_indices = np.array_split(range(len(train_dataset)), num_clients)
    train_dataset_partitioned = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in train_dataset_split_indices[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])
        train_dataset_partitioned.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_dataset_partitioned


def class_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    **kwargs,
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks in a
    non-IID manner by letting each client only have a subset of all classes.
    :param train_dataset: the training dataset
    :param num_clients: number of clients
    :param visualization: whether to visualize the data distribution among clients
    :param output_dirname: the directory to save the plot
    :param output_filename: the filename to save the plot
    :param seed: the random seed
    :return train_dataset_partitioned: a list of `torch.utils.data.Dataset` for each client
    """
    np.random.seed(seed)
    Cmin = {1: 10, 2: 7, 3: 6, 4: 5, 5: 5, 6: 4, 7: 4, "none": 3}
    Cmax = {1: 10, 2: 8, 3: 8, 4: 7, 5: 6, 6: 6, 7: 5, "none": 5}

    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()
    num_classes = len(labels)

    default_cmin = Cmin[num_clients] if num_clients in Cmin else Cmin["none"]
    default_cmax = Cmax[num_clients] if num_clients in Cmax else Cmax["none"]
    cmin = int(kwargs.get("min_classes_per_client", default_cmin))
    cmax = int(kwargs.get("max_classes_per_client", default_cmax))
    class_partition_alpha = kwargs.get("class_partition_alpha")

    if cmin < 1 or cmax < 1:
        raise ValueError("min_classes_per_client and max_classes_per_client must be positive.")
    if cmin > cmax:
        raise ValueError("min_classes_per_client cannot be greater than max_classes_per_client.")
    if cmax > num_classes:
        raise ValueError(
            f"max_classes_per_client={cmax} exceeds the number of classes ({num_classes})."
        )
    if class_partition_alpha is not None and float(class_partition_alpha) <= 0:
        raise ValueError("class_partition_alpha must be positive when provided.")

    while True:
        class_partition = {}
        client_classes = {}
        for i in range(num_clients):
            cnum = np.random.randint(cmin, cmax + 1)
            classes = np.random.permutation(range(num_classes))[:cnum]
            client_classes[i] = classes
            for cls in classes:
                class_partition[cls] = class_partition.get(cls, 0) + 1
        if len(class_partition) == num_classes:
            break

    partition_endpoints = {}
    for label in labels:
        total_size = len(label_indices[label])
        partitions = class_partition[label]
        if class_partition_alpha is None:
            partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))
            partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size
            partition_lengths = np.array(partition_lengths, dtype=np.int32)
            partition_lengths[-1] += total_size - int(np.sum(partition_lengths))
        else:
            base_lengths = np.ones(partitions, dtype=np.int32)
            remaining = total_size - partitions
            if remaining < 0:
                raise ValueError(
                    f"Class {label} has only {total_size} samples but is assigned to {partitions} clients."
                )
            proportions = np.random.dirichlet(
                np.full(partitions, float(class_partition_alpha), dtype=np.float64)
            )
            partition_lengths = base_lengths + np.random.multinomial(remaining, proportions)

        endpoints = np.cumsum(partition_lengths)
        endpoints = np.array(endpoints, dtype=np.int32)
        endpoints[-1] = total_size
        partition_endpoints[label] = endpoints

    partition_pointer = {label: 0 for label in labels}
    client_datasets = []
    client_dataset_info = {}
    for i in range(num_clients):
        client_dataset_info[i] = {}
        sample_indices = []
        client_class = client_classes[i]
        for cls in client_class:
            start_idx = (
                0
                if partition_pointer[cls] == 0
                else partition_endpoints[cls][partition_pointer[cls] - 1]
            )
            end_idx = partition_endpoints[cls][partition_pointer[cls]]
            sample_indices.extend(label_indices[cls][start_idx:end_idx])
            partition_pointer[cls] += 1
            client_dataset_info[i][cls] = end_idx - start_idx
        client_datasets.append(sample_indices)

    if visualization:
        classes_samples = [len(label_indices[label]) for label in labels]
        sample_matrix = np.zeros((len(classes_samples), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                sample_matrix[cls][i] = client_dataset_info[i][cls]
        plot_distribution(num_clients, classes_samples, sample_matrix, output_dirname, output_filename)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for idx in client_datasets[i]:
            train_data_input.append(train_dataset[idx][0].tolist())
            train_data_label.append(train_dataset[idx][1])

        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets


def dirichlet_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    alpha1: int = 8,
    alpha2: int = 0.5,
    seed: int = 42,
    **kwargs,
):
    """
    Partition a `torch.utils.data.Dataset` into `num_clients` clients chunks
    using two Dirichlet distributions: one for the number of elements for each client
    and the other for the number of elements from each class for each client.
    """
    np.random.seed(seed)
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    for label in labels:
        np.random.shuffle(label_indices[label])

    p1 = [1 / num_clients for _ in range(num_clients)]
    p2 = [len(label_indices[label]) for label in labels]
    p2 = [p / sum(p2) for p in p2]

    q1 = [alpha1 * i for i in p1]
    q2 = [alpha2 * i for i in p2]

    weights = np.random.dirichlet(q1)
    individuals = np.random.dirichlet(q2, num_clients)

    classes_samples = [len(label_indices[label]) for label in labels]

    normalized_portions = np.zeros(individuals.shape)
    for i in range(num_clients):
        for j in range(len(classes_samples)):
            normalized_portions[i][j] = (
                weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])
            )

    sample_matrix = np.multiply(np.array([classes_samples] * num_clients), normalized_portions).transpose()

    for i in range(len(classes_samples)):
        total = 0
        for j in range(num_clients - 1):
            sample_matrix[i][j] = int(sample_matrix[i][j])
            total += sample_matrix[i][j]
        sample_matrix[i][num_clients - 1] = classes_samples[i] - total

    num_elements = np.array(sample_matrix.transpose(), dtype=np.int32)
    client_totals = num_elements.sum(axis=1)
    empty_clients = np.where(client_totals == 0)[0]
    for empty_client in empty_clients:
        donor_totals = num_elements.sum(axis=1)
        donor_candidates = np.where(donor_totals > 1)[0]
        if len(donor_candidates) == 0:
            raise ValueError(
                "Dirichlet partition produced an empty client dataset and no donor client had enough samples to rebalance."
            )
        donor = donor_candidates[np.argmax(donor_totals[donor_candidates])]
        donor_classes = np.where(num_elements[donor] > 0)[0]
        donor_class = donor_classes[np.argmax(num_elements[donor][donor_classes])]
        num_elements[donor][donor_class] -= 1
        num_elements[empty_client][donor_class] += 1

    if visualization:
        plot_distribution(
            num_clients,
            classes_samples,
            num_elements.transpose(),
            output_dirname,
            output_filename,
        )

    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        train_data_input = []
        train_data_label = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i - 1][j]
            end = sum_elements[i][j]
            for idx in label_indices[label][start:end]:
                train_data_input.append(train_dataset[idx][0].tolist())
                train_data_label.append(train_dataset[idx][1])
        train_datasets.append(
            Dataset(
                torch.FloatTensor(train_data_input),
                torch.tensor(train_data_label),
            )
        )
    return train_datasets


def build_vision_transform(
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
):
    transform_steps = [transforms.ToTensor()]
    if normalize_mean is not None or normalize_std is not None:
        if normalize_mean is None or normalize_std is None:
            raise ValueError("normalize_mean and normalize_std must be provided together.")
        transform_steps.append(transforms.Normalize(tuple(normalize_mean), tuple(normalize_std)))
    return transforms.Compose(transform_steps)


def get_torchvision_dataset(
    dataset_cls,
    num_clients: int,
    client_id: int,
    partition_strategy: str = "iid",
    include_server_distribution: bool = False,
    load_validation_dataset: bool = True,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
    root_dir: Optional[str] = None,
    download: bool = True,
    **kwargs,
):
    """
    Load a torchvision image dataset and partition the training split for federated clients.
    """
    dataset_root = (
        os.path.join(os.getcwd(), "datasets", "RawData")
        if root_dir is None
        else root_dir
    )
    transform = build_vision_transform(
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    test_dataset = None
    if load_validation_dataset or include_server_distribution:
        test_data_raw = dataset_cls(dataset_root, download=download, train=False, transform=transform)

        test_data_input = []
        test_data_label = []
        for idx in range(len(test_data_raw)):
            test_data_input.append(test_data_raw[idx][0].tolist())
            test_data_label.append(test_data_raw[idx][1])
        test_dataset = Dataset(torch.FloatTensor(test_data_input), torch.tensor(test_data_label))

    train_data_raw = dataset_cls(dataset_root, download=download, train=True, transform=transform)

    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_nomiid":
        train_datasets = dirichlet_noniid_partition(train_data_raw, num_clients, **kwargs)
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    if kwargs.get("visualization", False) and include_server_distribution and test_dataset is not None:
        plot_distribution_with_server(
            train_datasets,
            test_dataset,
            kwargs.get("output_dirname"),
            kwargs.get("output_filename"),
        )

    return train_datasets[client_id], test_dataset
