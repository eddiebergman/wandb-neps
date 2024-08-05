from __future__ import annotations

import threading
import wandb
from wandb.util import auto_project_name
import time
import os
from wandb.apis import InternalApi
from wandb import wandb_sdk
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb import controller as wandb_controller, wandb_agent


_api: InternalApi | None = None


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
    sweep_id: str, stop_event: threading.Event, verbose: bool = True
) -> None:
    tuner = wandb_controller(sweep_id)
    tuner._start_if_not_started()
    while not tuner.done():
        if verbose:
            print_status = True
            print_actions = True
            print_debug = True

        if print_status:
            tuner.print_status()
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
    wandb.termlog("Starting wandb agent 🕵️")
    wandb_agent.agent(sweep_id, entity=entity, project=project, count=count)


def sweep(
    config_yaml_or_sweep_id: str,
    count: int = 1,
    prior_runs: list[str] | None = [],
):
    config_yaml = config_yaml_or_sweep_id

    api = _get_cling_api()
    if not api.is_authenticated:
        raise Exception("Login to W&B to use the sweep feature")

    sweep_obj_id = None
    action = "Updating" if sweep_obj_id else "Creating"
    wandb.termlog(f"{action} sweep from: {config_yaml}")
    config = sweep_utils.load_sweep_config(config_yaml)
    assert config is not None
    wandb.termlog(f"Loaded config: {config}")

    config.setdefault("controller", {})
    config["controller"]["type"] = "local"

    tuner = wandb_controller()
    err = tuner._validate(config)
    if err:
        wandb.termerror(f"Error in sweep file: {err}")
        return

    env = os.environ
    entity = env.get("WANDB_ENTITY") or config.get("entity") or api.settings("entity")
    project = (
        env.get("WANDB_PROJECT")
        or config.get("project")
        or api.settings("project")
        or auto_project_name(config.get("program"))
    )

    sweep_id, warnings = api.upsert_sweep(
        config,
        project=project,
        entity=entity,
        obj_id=sweep_obj_id,
        prior_runs=prior_runs,
    )
    sweep_utils.handle_sweep_config_violations(warnings)

    # Log nicely formatted sweep information
    wandb.termlog(f"sweep with ID: {sweep_id}")

    sweep_url = wandb_sdk.wandb_sweep._get_sweep_url(api, sweep_id)
    if sweep_url:
        wandb.termlog(f"View sweep at: {sweep_url}")

    # re-probe entity and project if it was auto-detected by upsert_sweep
    entity = entity or env.get("WANDB_ENTITY")
    project = project or env.get("WANDB_PROJECT")

    if entity and project:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    elif project:
        sweep_path = f"{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    if sweep_path.find(" ") >= 0:
        sweep_path = f"{sweep_path!r}"

    wandb.termlog(f"Run sweep agent with: {sweep_path}")
    wandb.termlog("Starting wandb controller...")

    # Create threads
    stop_controller = threading.Event()
    controller_thread = threading.Thread(
        target=run_controller, args=(sweep_id, stop_controller, True)
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
    args = parser.parse_args()
    sweep(
        config_yaml_or_sweep_id=args.sweep,
        prior_runs=[],
        count=args.count,
    )
