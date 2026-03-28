# StoreGate

A multi-backend data management library for deep learning workflows.

StoreGate provides a unified interface for storing, retrieving, and managing
training data across in-memory and disk-backed backends, with built-in
PyTorch integration and hyperparameter search support.

## Features

- **Multi-backend storage** -- switch between in-memory (`NumpyDatabase`) and
  disk-persistent (`ZarrDatabase`) backends without changing application code.
- **Subscript syntax** -- read and write data with `sg["train"]["x"][0:10]`.
- **PyTorch integration** -- compile datasets and dataloaders directly from
  stored data via `PytorchTask`.
- **Hyperparameter search** -- grid search and random search agents with
  multi-GPU parallel execution.
- **Transactional writes** -- `staged_add` ensures atomic multi-variable
  inserts with automatic rollback on failure.
- **Cross-backend transfer** -- stream data between backends with
  `transfer_data`.

## Requirements

- Python >= 3.12
- numpy >= 2.4.2
- zarr >= 3.1.5
- tqdm >= 4.0

Optional (for PyTorch features):

- torch >= 2.0
- torchinfo >= 1.8

## Installation

```bash
pip install storegate

# With PyTorch support
pip install storegate[torch]

# Development
pip install storegate[dev]
```

## Quick Start

### Basic usage with NumpyDatabase

```python
import numpy as np
from storegate import StoreGate, NumpyDatabase

sg = StoreGate(
    backends={"mem": NumpyDatabase()},
    backend="mem",
    data_id="experiment_1",
)

with sg:
    # Store training data
    sg.add_data("train", "x", np.random.randn(1000, 16).astype(np.float32))
    sg.add_data("train", "y", np.random.randn(1000, 1).astype(np.float32))

    # Append more data
    sg.add_data("train", "x", np.random.randn(500, 16).astype(np.float32))
    sg.add_data("train", "y", np.random.randn(500, 1).astype(np.float32))

    # Check alignment
    report = sg.compile()
    print(report["is_compiled"])  # True -- all variables have 1500 events

    # Read with subscript syntax
    batch = sg["train"]["x"][0:32]        # shape (32, 16)
    single = sg["train"]["y"][0]          # shape (1,)

    # Update in-place
    sg["train"]["x"][0] = np.zeros(16, dtype=np.float32)

    # Display summary
    sg.show_info()
```

### Disk-persistent storage with ZarrDatabase

```python
from storegate import StoreGate, NumpyDatabase, ZarrDatabase

sg = StoreGate(
    backends={
        "mem": NumpyDatabase(),
        "disk": ZarrDatabase("./data/store", mode="a"),
    },
    backend="mem",
    data_id="run_01",
)

with sg:
    # Work in memory first
    sg.add_data("train", "x", x_train)
    sg.add_data("train", "y", y_train)

    # Transfer to disk
    sg.disk.set_data_id("run_01")
    sg.transfer_data("mem", "disk", "train", "x")
    sg.transfer_data("mem", "disk", "train", "y")
```

To read an existing store:

```python
sg = StoreGate(
    backends={"disk": ZarrDatabase("./data/store", mode="r")},
    backend="disk",
    data_id="run_01",
)
with sg:
    data = sg.get_data("train", "x")
```

### PyTorch training

```python
import torch.nn as nn
import torch.optim as optim
from storegate.task import PytorchTask

task = PytorchTask(
    phases={"train": "train", "valid": "valid", "test": "test"},
    var_names={"inputs": "x", "outputs": "pred", "true": "y"},
    model=nn.Linear,
    model_args={"in_features": 16, "out_features": 1},
    optimizer=optim.Adam,
    optimizer_args={"lr": 1e-3},
    loss=nn.MSELoss,
    num_epochs=10,
    metrics=["acc"],
    dataloader_args={"batch_size": 64, "shuffle": True},
    progress_bar=True,
)
task.storegate = sg
result = task.execute()
# result == {"fit": {"train": [...], "valid": [...]}, "predict": {...}}
```

### Hyperparameter search

```python
from storegate.agent import GridSearchAgent

agent = GridSearchAgent(
    storegate=sg,
    task=PytorchTask,
    task_args={
        "phases": {"train": "train", "test": "test"},
        "var_names": {"inputs": "x", "true": "y"},
        "model": nn.Linear,
        "model_args": {"in_features": 16, "out_features": 1},
        "optimizer": optim.Adam,
        "loss": nn.MSELoss,
        "num_epochs": 5,
    },
    hps={
        "optimizer__lr": [1e-2, 1e-3, 1e-4],
        "num_epochs": [5, 10],
    },
    num_trials=3,
)

results = agent.execute()
agent.save_results(results, "search_results.json")
```

For multi-GPU parallel search, pass `cuda_ids`:

```python
agent = GridSearchAgent(
    ...,
    cuda_ids=[0, 1, 2, 3],  # 4 jobs in parallel
)
```

## Core Concepts

### Data hierarchy

```
StoreGate
 └── backend (e.g. "mem", "disk")
      └── data_id (e.g. "experiment_1")
           └── phase (e.g. "train", "valid", "test")
                └── var_name (e.g. "x", "y", "pred")
                     └── numpy.ndarray (num_events, *features)
```

All arrays are indexed along axis 0 (the event axis).
Variables within the same phase must share the same event count once
compiled.

### Backends

| Backend | Persistence | Mode | Use case |
|---|---|---|---|
| `NumpyDatabase` | In-memory | Always writable | Fast prototyping, ephemeral data |
| `ZarrDatabase` | Disk (Zarr V3) | `r`, `r+`, `a`, `w-` | Large datasets, persistence |

### Backend views

Access a specific backend through attribute syntax:

```python
sg.mem.add_data("train", "x", data)   # writes to "mem" backend
sg.disk.get_data("train", "x")        # reads from "disk" backend
sg.all.set_phase("train")             # broadcasts to all backends
```

### Read-only views

Pin a snapshot for safe concurrent reads (e.g. in a DataLoader):

```python
view = sg.pin(backend="mem", data_id="run_01")
view["train"]["x"][0:32]  # read-only access
view.is_writable()         # False
```

### Transactional writes

`staged_add` writes multiple variables atomically:

```python
with sg.staged_add("train", ["x", "y"]) as tx:
    tx.add_data("x", x_data)
    tx.add_data("y", y_data)
# Both variables become visible only after the block succeeds.
# On exception, staged data is cleaned up automatically.
```

### Streaming

Read large variables in chunks without loading the full array:

```python
for chunk in sg.stream_data("train", "x"):
    process(chunk)
```

`ZarrDatabase` yields storage-aligned chunks; `NumpyDatabase` yields one
full-array snapshot.

## API Reference

### StoreGate

| Method | Description |
|---|---|
| `__init__(backends, backend, data_id)` | Create with optional backends dict, active backend, and data_id |
| `register_backend(name, db)` | Register a new backend |
| `set_backend(name)` / `get_backend()` | Switch / query the active backend |
| `using_backend(name)` | Context manager for temporary backend switch |
| `set_data_id(id)` / `get_data_id()` | Set / get the active data_id |
| `set_phase(phase)` / `delete_phase(phase)` | Create / delete a phase |
| `add_data(phase, var, data)` | Append data (creates phase if needed) |
| `update_data(phase, var, data, index)` | Replace data at index |
| `get_data(phase, var, index)` | Retrieve data (defensive copy) |
| `delete_data(phase, var)` | Delete a variable |
| `stream_data(phase, var)` | Yield data in chunks |
| `copy_data(phase, var, new_var)` | Copy a variable within the same phase |
| `rename_data(phase, var, new_var)` | Rename a variable |
| `staged_add(phase, var_names)` | Transactional multi-variable write |
| `transfer_data(src, dst, phase, var)` | Cross-backend copy via streaming |
| `compile()` | Check event-count alignment |
| `pin(backend, data_id)` | Create a read-only snapshot view |
| `show_info(file, compile)` | Print a formatted summary table |

### ZarrDatabase modes

| Mode | Behavior |
|---|---|
| `"r"` | Read-only. Store must exist. |
| `"r+"` | Read-write. Store must exist. |
| `"a"` | Append. Creates store if missing. |
| `"w-"` | Create new. Fails if store exists. |

### PytorchTask

| Parameter | Type | Description |
|---|---|---|
| `phases` | `str \| dict` | Phase mapping (`{"train": "train", "test": "test"}`) |
| `var_names` | `dict` | `{"inputs": ..., "outputs": ..., "true": ...}` |
| `model` | class/str/instance | Model to train |
| `model_args` | `dict` | Arguments passed to model constructor |
| `optimizer` | class/str | Optimizer class |
| `optimizer_args` | `dict` | Arguments passed to optimizer (excluding `params`) |
| `loss` | class/str/instance | Loss function |
| `num_epochs` | `int` | Number of training epochs |
| `metrics` | `list \| dict` | `["acc"]`, `["binary_acc"]`, `["multi_acc"]`, or callables |
| `device` | `str` | `"auto"`, `"cpu"`, or `"cuda:0"` |
| `dataloader_args` | `dict` | Passed to `torch.utils.data.DataLoader` |
| `progress_bar` | `bool` | Show tqdm progress bar |

Hyperparameters are set with namespace prefixes:
`"model__hidden_size"`, `"optimizer__lr"`, `"loss__reduction"`,
`"dataset__preload"`, `"dataloader__batch_size"`.

### SearchAgent

| Parameter | Type | Description |
|---|---|---|
| `storegate` | `StoreGate` | Shared data store |
| `task` | `type` | Task class (not an instance) |
| `task_args` | `dict` | Arguments passed to task constructor |
| `hps` | `dict[str, list]` | Hyperparameter search space |
| `num_trials` | `int` | Repetitions per combination |
| `cuda_ids` | `list[int] \| None` | GPU IDs for parallel execution |

`GridSearchAgent` evaluates the Cartesian product of all HP values.
`RandomSearchAgent` samples `num_samples` random combinations.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
