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

## No early_terminate
```bash
python sweep-starter.py \
    [--entity <str>] \
    [--project <str>] \
    [--count <int>] \
    [--sweeper-refresh-rate <float>] \
    [--trainer-epoch-sleep-duration <float>]
```

For example, you can run the following command which should be sufficient for testing.
```bash
python sweep-starter.py \
    --entity eddiebergmanhs \
    --project sweep-test \
    --count 15 \
    --sweeper-refresh-rate 1 \
    --trainer-epoch-sleep-duration 0.5
```

## With early terminate
It seems there's quite some delay in:

1. The sweep agent getting the most up to date history and issuing a stop command
2. The worker agent recieving the stop command and cancelling the run.

By setting the `trainer-epoch-sleep-duration`, this gives more time for this synchronization to occur
but it makes the entire testing process terminally slow...

The example below sets the count for 50 runs just to get a nice overview...
```bash
python sweep-starter.py \
    --entity eddiebergmanhs \
    --project sweep-test-with-hb \
    --count 50 \
    --sweeper-refresh-rate 5 \
    --trainer-epoch-sleep-duration 5
```
