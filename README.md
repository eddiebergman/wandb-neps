# NePS Sweeps Example
Proof of concept for using NePS with W&B sweeps.

The two files used are `sweep-starter.py` which spawns both sweep-agent and evaluating agent
and `train.py` as the actual training procedure _i.e._ user code.

Please read the `sweep-starter.py` file for more information on implementation details.

## Setup
```bash
# Clone this repo
# cd into it
# Make a venv
mkdir -p vendored
git clone git@github.com:automl/neps.git vendored/neps
cd vendored/neps
git checkout 7aceeaa
cd ../../

pip install "sweeps==0.2.0" "wandb==0.17.7" ./vendored/neps

# NePS uses torch, if you want a CPU version:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run no early-stopping
The following will run a faily quick demo, with no early stopping enabled.

```bash
# Replace with your entity and project
python sweep-starter.py --entity eddiebergmanhs --project sweep-test
```

## With early-stopping
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
    --include-early-stopping \
    --trainer-epoch-sleep-duration 5
```

## Extra args:
```bash
--count <int> # Number of "epochs" (Default: 15)
--sweeper-refresh-rate <float> # How long sweeper agent should sleep before its loop (Defeault: 1)
```
