from __future__ import annotations

import threading
from typing import Any
import wandb
import time
from wandb.apis import InternalApi
from wandb import wandb_sdk
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb import controller as wandb_controller, wandb_agent
from wandb.wandb_controller import _WandbController
import sweeps
import sweeps.params


_api: InternalApi | None = None

ENTITY = "eddiebergmanhs"
PROJECT = "sweep-test"


# NOTE: Signature of a function that implements _custom_search
def my_next_run(
    sweep_config: sweeps.SweepConfig,
    runs: list[sweeps.SweepRun],
    validate: bool = False,
    **kwargs: Any,
) -> sweeps.SweepRun | None:
    return random_search_next_runs(sweep_config)[0]


def random_search_next_runs(sweep_config: sweeps.SweepConfig) -> list[sweeps.SweepRun]:
    print("MY RANDOM SEARCH")
    if sweep_config["method"] != "random":
        raise ValueError("Invalid sweep configuration for random_search_next_run.")
    params = sweeps.params.HyperParameterSet.from_config(sweep_config["parameters"])

    for param in params:
        param.value = param.sample()

    run = sweeps.SweepRun(config=params.to_config())
    return [run]


def _get_cling_api(reset=None):
    """Get a reference to the internal api with cling settings."""
    global _api
    if reset:
        _api = None
        wandb_sdk.wandb_setup._setup(_reset=True)
    if _api is None:
        # TODO(jhr): make a settings object that is better for non runs.
        # only override the necessary setting
        wandb.setup(settings=dict(_cli_only_mode=True))
        _api = InternalApi()
    return _api


def run_controller(
    tuner: _WandbController,
    stop_event: threading.Event,
    verbose: bool = True,
) -> None:
    tuner._start_if_not_started()
    while not tuner.done():
        if verbose:
            print_status = True
            print_actions = True
            print_debug = True

        if print_status:
            tuner.print_status()

        # This is the entry point
        tuner.step()

        if print_actions:
            tuner.print_actions()
        if print_debug:
            tuner.print_debug()

        if stop_event.is_set():
            wandb.termlog("Stopping wandb controller as agent is done...")
            break

        time.sleep(1)


def run_agent(sweep_id: str, entity: str, project: str, count: int) -> None:
    wandb.termlog("Starting wandb agent")
    wandb_agent.agent(sweep_id, entity=entity, project=project, count=count)


def sweep(
    config_yaml_or_sweep_id: str,
    count: int = 1,
    entity: str = ENTITY,
    project: str = PROJECT,
    prior_runs: list[str] | None = [],
):
    config_yaml = config_yaml_or_sweep_id

    api = _get_cling_api()
    if not api.is_authenticated:
        raise Exception("Login to W&B to use the sweep feature")

    wandb.termlog(f"Creating sweep from: {config_yaml}")
    config = sweep_utils.load_sweep_config(config_yaml)
    assert config is not None
    wandb.termlog(f"Loaded config: {config}")

    config.setdefault("controller", {})
    config["controller"]["type"] = "local"
    config["_custom_search"] = my_next_run

    tuner = wandb_controller(sweep_id_or_config=config, entity=entity, project=project)
    sweep_id = tuner.sweep_id

    print("-----------")
    print(sweep_id)
    wandb.termlog(f"sweep with ID: {sweep_id}")
    print("-----------")

    # DEBUG
    sweep_url = wandb_sdk.wandb_sweep._get_sweep_url(api, sweep_id)
    if sweep_url:
        wandb.termlog(f"View sweep at: {sweep_url}")

    wandb.termlog("Starting wandb controller...")

    # Create threads
    print(config)
    stop_controller = threading.Event()
    controller_thread = threading.Thread(
        target=run_controller, args=(tuner, stop_controller, True)
    )
    agent_thread = threading.Thread(
        target=run_agent, args=(sweep_id, entity, project, count)
    )

    # Start threads
    controller_thread.start()
    agent_thread.start()
    agent_thread.join()
    stop_controller.set()
    controller_thread.join()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sweep", type=str, default="sweep.yaml")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--project", type=str, default=PROJECT)
    args = parser.parse_args()
    sweep(
        config_yaml_or_sweep_id=args.sweep,
        prior_runs=[],
        count=args.count,
        entity=args.entity,
        project=args.project,
    )
