# Setup
```bash
# Make a venv
mkdir -p vendored
git clone git@github.com:wandb/wandb.git vendored/wandb
git clone git@github.com:wandb/sweeps.git vendored/sweeps

pip install -e vendored/wandb
# Don't ask ...
pip install -e vendored/sweeps --config-settings editable_mode=strict
```

#
Create a yaml file with the search space
```yaml
# config.yaml
program: train.py  # <- Note, this is the file you want to run
method: random
name: sweep
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size:
    values: [16,32,64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

Have a python file with the following at minimum:

```python
def main():
    run = wandb.init()

    # Note that we define values from `wandb.config`
    # instead of  defining hard values
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    # Your code to evaluate here
    # ...

    wandb.log({"val_acc": 0.5})

if __name__ == "__main__":
    main()
```

Start the loop

```bash
python sweep-starter.py
```
