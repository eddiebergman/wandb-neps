# Setup
```bash
# Make a venv
# ...

mkdir -p vendored
git clone git@github.com:wandb/wandb.git vendored/wandb
git clone git@github.com:wandb/sweeps.git vendored/sweeps
git clone git@github.com:automl/neps.git vendored/neps

pip install -e vendored/wandb
# Don't ask ...
pip install -e vendored/sweeps --config-settings editable_mode=strict

pip install -e vendored/neps

# If you want a CPU version of torch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run
The following command will spin up a sweep-agent in one thread, wait 2 seconds
and then spin up an agent (worker agent?) in another thread.

```bash
python sweep-starter.py [--entity <entity>] [--project <project>] [--count <count>]
```

For example
```bash
python sweep-starter.py [--entity <entity>] [--project <project>] [--count <count>]
```
