# storegate

A data management library for deep learning pipelines. StoreGate provides a unified interface for storing and retrieving ML experiment data across **train / valid / test** phases, backed by a hybrid zarr (disk) + numpy (memory) architecture.

---

## Installation

```bash
pip install storegate
```

With PyTorch support:

```bash
pip install "storegate[pytorch]"
```

**Requirements:** Python >= 3.12 | **Platform:** Linux

> **Note — Platform support:**
> StoreGate is developed and tested on **Linux only**.
> Some features depend on POSIX-specific APIs (e.g. `signal.SIGALRM` for
> serial job timeouts in `SearchAgent`). macOS may work for core
> functionality but is not officially tested. Windows is **not supported**.

---

## Quick Start

### 1. Store and retrieve data

```python
import numpy as np
from storegate import StoreGate

# Open in write mode ('w' creates/overwrites, 'a' appends, 'r' read-only)
sg = StoreGate(output_dir='./my_data', mode='w')
sg.set_data_id('experiment_01')

# Add data per phase
sg.add_data('x', x_train, phase='train')
sg.add_data('x', x_valid, phase='valid')
sg.add_data('x', x_test,  phase='test')

# Or add all splits in one call
sg.add_data_splits('y', train=y_train, valid=y_valid, test=y_test)

# Validate that all variables in each phase have consistent event counts
sg.compile(show_info=True)

# Retrieve data
x = sg.get_data('x', phase='train')          # all samples
x0 = sg.get_data('x', phase='train', index=0)  # single sample
```

### 2. Chained bracket access

```python
sg['train']['x'][:]      # all train samples for 'x'
sg['train']['x'][0]      # first sample
sg['train']['x'][0] = v  # update in-place
del sg['train']['x']     # delete variable

for var_name, data in sg['train'].items():
    print(var_name, data.shape)
```

### 3. Switch backends

Data can live on disk (zarr) or in memory (numpy). The active backend is switched with a context manager:

```python
sg.set_backend('zarr')    # default: reads/writes from disk
sg.set_backend('numpy')   # reads/writes from memory

# Temporarily switch backend
with sg.using_backend('numpy'):
    sg.add_data('x_cache', x_train, phase='train')

# Copy between backends
sg.copy_to_memory('x', phase='train')    # zarr → numpy
sg.copy_to_storage('x', phase='train')  # numpy → zarr
```

### 4. Re-open an existing store

```python
sg = StoreGate(output_dir='./my_data', mode='r', data_id='experiment_01')
x_train = sg.get_data('x', phase='train')
```

If the store was previously compiled and not modified afterwards, phase sizes are
restored automatically for the zarr backend, so `len(sg['train'])` works
immediately after reopening. Call `sg.compile()` again only if the store was
never compiled, or if data changed after the last successful compile.

---

## PyTorch Training

Subclass `PytorchTask` to define a training pipeline. The task wires your model, optimizer, and loss directly to StoreGate data.

```python
import torch.nn as nn
from storegate.task import PytorchTask

class MyTask(PytorchTask):
    pass  # override fit()/predict()/step_batch() as needed

task = MyTask(
    storegate=sg,
    input_var_names=['x'],
    true_var_names=['y'],
    output_var_names=['pred'],
    model=nn.Linear,
    model_args={'in_features': 784, 'out_features': 10},
    optimizer='Adam',
    optimizer_args={'lr': 1e-3},
    loss='CrossEntropyLoss',
    num_epochs=10,
    batch_size=64,
    metrics=['loss', 'acc'],
    device='auto',   # 'auto' selects CUDA if available, else CPU
)

task.execute()  # runs fit() then predict(), writes outputs to storegate
```

---

## Hyperparameter Search

### Grid search

Exhaustively evaluates all combinations of the given hyperparameter lists.

```python
from storegate.agent import GridSearchAgent

agent = GridSearchAgent(
    task=task,
    hps={
        'optimizer__lr':      [1e-3, 1e-4],
        'model__num_filters': [32, 64],
        'batch_size':         [64, 128],
    },
    cuda_ids=[0, 1],       # injects cuda_id values into jobs in submission order
    # default suffix_job_id=True writes test outputs as e.g.
    # pred_job0_trial0, pred_job1_trial0, ...
    json_dump='results.json',
)
agent.execute()
agent.finalize()
```

### Random search

Randomly samples `num_iter` combinations from the search space.

```python
from storegate.agent import RandomSearchAgent

agent = RandomSearchAgent(
    task=task,
    hps={
        'optimizer__lr':   [1e-3, 3e-4, 1e-4],
        'batch_size':      [32, 64, 128],
    },
    num_iter=10,
    seed=42,
    replace=False,      # default: unique combinations only
    cuda_ids=[0],
    json_dump='results.json',
)
agent.execute()
agent.finalize()
```

`replace=False` is the default and samples unique hyperparameter combinations
without replacement. Set `replace=True` to allow duplicate combinations.

`cuda_ids=None` runs jobs serially in the current process using the task's own device.
When `cuda_ids` is a list, those values control which `cuda_id` is injected into each job's hyperparameters.
Jobs are assigned IDs in submission order; this is not an exclusive worker-to-GPU binding.
By default, `suffix_job_id=True`, so `output_var_names` are suffixed with `_job{job_id}_trial{trial_id}` to avoid collisions between parallel jobs and repeated trials. When `num_trials` is not set, the implicit single trial uses `trial0`. Set `suffix_job_id=False` only when each job writes to a different `data_id` or otherwise guarantees isolated output variable names.

**Hyperparameter key conventions:**

| Prefix | Target |
|--------|--------|
| `model__<name>` | `model_args[name]` |
| `optimizer__<name>` | `optimizer_args[name]` |
| `loss__<name>` | `loss_args[name]` |
| `<name>` (no prefix) | task attribute `_<name>` (e.g. `batch_size`, `num_epochs`) |

---

## Logging

```python
from storegate import logger

logger.set_level(logger.DEBUG)           # DEBUG / INFO / WARN / ERROR / DISABLED
logger.add_file_handler('run.log')       # also write logs to file
```

---

## License

See [LICENSE](LICENSE).
