# Setup
```bash
# Make a venv
mkdir -p vendored
git clone git@github.com:automl/neps.git vendored/neps
cd vendored/neps
git checkout 7aceeaa

pip install "sweeps==0.2.0" "wandb==0.17.7" ./vendored/neps

# NePS uses torch, if you want a CPU version:
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
# Replace with your entity and project
python sweep-starter.py --entity eddiebergmanhs --project sweep-test
```

## With early Terminate
To use W&B existing early termination, use the following command:

> [!WARNING] As the communication between sweep-agent <-> W&B servers <-> worker agent is not instant,
> we have to introduce quite a bit of sleeping, to allow the worker agent to be terminated mid run.
> This makes the example much slower, time to get a coffee, fix a bug, reply to an issue or get some food.
> If any runs are stopped, a `stopping.log` file is created

```bash
# Replace with your entity and project
python sweep-starter.py \
    --entity eddiebergmanhs \
    --project sweep-test \
    --include-early-terminate \
    --trainer-epoch-sleep-duration 5
```

## Extra args:
```bash
--count <int> # Number of "epochs" (Default: 15)
--sweeper-refresh-rate <float> # How long sweeper agent should sleep before its loop (Defeault: 1)
```
