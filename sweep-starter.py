"""This script provides a prototype of how to use the W&B sweeps feature with NEPS.

It tries to provide as much of the control as possible in one script, i.e. it
spins up the 'custom' sweep agent and the 'local' worker agent, both in seperate threads.

The main thing to make this work was to set the `method` to `custom` in the sweep config
and to provide a custom search function `configure_search(my_next_run)` to the controller.

There are comments provided through the rest of the script
"""

from __future__ import annotations

import logging
import time
import threading
import numpy as np
from collections.abc import Mapping
from typing import Any

import wandb
from wandb.apis import InternalApi
from wandb import wandb_sdk
from wandb import controller as wandb_controller, wandb_agent
from wandb.wandb_controller import _WandbController

import sweeps
import sweeps.params

import neps
from neps.optimizers.base_optimizer import SampledConfig
from neps.search_spaces.parameter import Parameter
from neps.search_spaces.search_space import SearchSpace
from neps.state.trial import Trial
from neps.optimizers.bayesian_optimization.optimizer import BayesianOptimization

from sweeps.params import HyperParameter

# DEBUG on W&B agent was too noisy for this demo
logging.getLogger("wandb.wandb_agent").setLevel(logging.INFO)

STOPPING_LOG_FILE = "stopping.log"

_api: InternalApi | None = None

ENTITY = "eddiebergmanhs"
PROJECT = "sweep-test"
SWEEP_CONFIG = {
    "program": "train.py",
    "controller": {"type": "local"},
    "method": "custom",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "lr": {"min": 0.0001, "max": 0.1},
        "epochs": {"value": 15},
    },
}

STATE_MAPPING: Mapping[sweeps.run.RunState, Trial.State | None] = {
    sweeps.run.RunState.pending: Trial.State.PENDING,
    sweeps.run.RunState.running: Trial.State.EVALUATING,
    sweeps.run.RunState.finished: Trial.State.SUCCESS,
    sweeps.run.RunState.killed: Trial.State.CRASHED,  # NOTE: We don't distinguish between killed and crashed
    sweeps.run.RunState.crashed: Trial.State.CRASHED,
    sweeps.run.RunState.failed: Trial.State.FAILED,
    # TODO(eddiebergman): I don't know what these mean yet...
    sweeps.run.RunState.preempted: None,
    sweeps.run.RunState.preempting: None,
}


def parse_hp(hp: sweeps.params.HyperParameter) -> Parameter:
    # Python 3.10 match statement would be nice right about now...
    if hp.type == HyperParameter.CONSTANT:
        assert hp.value is not None
        return neps.ConstantParameter(hp.value)
    if hp.type == HyperParameter.CATEGORICAL:
        return neps.CategoricalParameter(hp.config["values"])
    if hp.type == HyperParameter.INT_UNIFORM:
        return neps.Integer(hp.config["min"], hp.config["max"])
    if hp.type == HyperParameter.UNIFORM:
        return neps.Float(hp.config["min"], hp.config["max"])
    if hp.type == HyperParameter.LOG_UNIFORM_V2:
        return neps.Float(hp.config["min"], hp.config["max"], log=True)

    # NOTE: We will add support for all of these soon
    if hp.type == HyperParameter.CATEGORICAL_PROB:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.LOG_UNIFORM_V1:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.INV_LOG_UNIFORM_V1:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.INV_LOG_UNIFORM_V2:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_UNIFORM:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_LOG_UNIFORM_V1:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_LOG_UNIFORM_V2:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.NORMAL:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_NORMAL:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.LOG_NORMAL:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_LOG_NORMAL:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.BETA:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")
    if hp.type == HyperParameter.Q_BETA:
        raise NotImplementedError(f"{hp.type}: {hp.config} is not supported yet")

    raise NotImplementedError(f"{hp.type}: {hp.config} is unknown")


def parse_searchspace(
    parameter_dict: dict[str, Any],
) -> SearchSpace:
    wandb_hp_set = sweeps.params.HyperParameterSet.from_config(parameter_dict)
    return SearchSpace(**{hp.name: parse_hp(hp) for hp in wandb_hp_set})


def sweep_run_to_trial(
    run: sweeps.SweepRun, metric_to_optimize: str, minimize: bool
) -> Trial:
    state = STATE_MAPPING[run.state]
    if state is None:
        raise NotImplementedError(f"Unknown how to handle run state: {run.state}")

    if run.name is None:
        raise NotImplementedError(f"Not sure how to handle run without name yet: {run}")

    if not sweeps.run.run_state_is_terminal(run.state):
        report = None
    else:
        if run.summary_metrics is not None:
            # NOTE(eddiebergman): This is not the last loss value reported but rather the best loss
            # seen. Perhaps this should be `learning_curve[-1]`?
            metric = run.metric_extremum(
                metric_to_optimize,
                kind="minimize" if minimize else "maximize",
            )

            metric_history = run.metric_history(metric_to_optimize)
            if not minimize:
                loss = -float(metric)
                learning_curve = [-float(v) for v in metric_history]
            else:
                loss = float(metric)
                learning_curve = [float(v) for v in metric_history]

            report = Trial.Report(
                trial_id=run.name,
                learning_curve=learning_curve,
                loss=loss,
                cost=None,  # Not definiable through W&B?
                # -------- Not needed for optimization --------
                extra={},
                evaluation_duration=-1,
                err=None,
                tb=None,
                reported_as=(
                    "success"
                    if state == Trial.State.SUCCESS
                    else "failed"
                    if state == Trial.State.FAILED
                    else "crashed"
                ),
            )
        else:
            report = Trial.Report(
                trial_id=run.name,
                learning_curve=None,
                loss=None,
                cost=None,  # Not definiable through W&B
                # -------- Not needed for optimization --------
                extra={},
                evaluation_duration=-1,
                err=None,
                tb=None,
                reported_as="crashed",  # TODO(eddiebergman): Only reason to have no `summary_metrics` is crash?
            )

    return Trial(
        config={
            k: d["value"]
            for k, d in run.config.items()
            # NOTE(eddiebergman): I have no idea how this get's into the `run.config` but it's there.
            # Seems to be kind telemetry
            if k != "_wandb"
        },
        state=state,
        report=report,
        metadata=Trial.MetaData(
            id=run.name,
            previous_trial_id=(
                run.search_info.get("previous_config_id", None)
                if run.search_info is not None
                else None
            ),
            # ------- Not used by optimizer for anything meaningful -------
            sampling_worker_id="",  # Not used here
            time_sampled=-1,  # Not used here
            location="",  # Not used here
            previous_trial_location=None,  # Not used here
        ),
    )


def sampled_trial_to_sweep_run(sampled_config: SampledConfig) -> sweeps.SweepRun:
    # NOTE(eddiebergman): W&B serialize values with which means we can only have
    # JSON serializable primitives. All of what we need for basic optimization
    # works fine when serialized, except our seeding mechanism (grrrr torch's rng system).
    # Not an issue in the W&B context, although we should change our mechanism in NePS.

    def as_native_python_type_for_serialization(
        v: int | float | np.number,
    ) -> int | float:
        if isinstance(v, np.number):
            return v.item()
        return v

    return sweeps.SweepRun(
        # NOTE(eddiebergman): Found a comment in `sweeps/params.py`
        #
        # > Because of historical reason the first level of nesting requires "value" key
        #
        config={
            k: {"value": as_native_python_type_for_serialization(v)}
            for k, v in sampled_config.config.items()
        },
        name=sampled_config.id,
        # TODO(eddiebergman): Remove this
        search_info={"previous_config_id": sampled_config.previous_config_id},
    )


def my_next_run(
    sweep_config: sweeps.SweepConfig,
    runs: list[sweeps.SweepRun],
) -> sweeps.SweepRun | None:
    metric_def = sweep_config.get("metric", None)
    assert metric_def is not None

    # Goal is either "minimize" or "maximize"
    goal = metric_def.get("goal", None)
    if goal is None:
        raise ValueError(f"Metric definition must have a goal. {metric_def}")

    assert goal.lower() in ("minimize", "maximize")
    minimize = goal.lower() == "minimize"

    metric_to_optimize_name = metric_def.get("name", None)
    if metric_to_optimize_name is None:
        raise ValueError(f"Metric definition must have a name. {metric_def}")

    search_space = parse_searchspace(sweep_config["parameters"])
    opt = BayesianOptimization(pipeline_space=search_space)
    trials = [
        sweep_run_to_trial(
            run,
            metric_to_optimize=metric_to_optimize_name,
            minimize=minimize,
        )
        for run in runs
    ]
    next_suggestion, _ = opt.ask(
        trials={t.metadata.id: t for t in trials},
        budget_info=None,
        optimizer_state={},
    )
    return sampled_trial_to_sweep_run(next_suggestion)


# Vendored out since it's used in demo script
def _get_client_api(reset=None):
    """Get a reference to the internal api with client settings."""
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


# A custom step function as the one in W&B sweeps seemed to be continously sampling
def custom_step(tuner: _WandbController) -> None:
    tuner._step()

    # only schedule one run at a time (for now)
    if tuner._controller and tuner._controller.get("schedule"):
        pass
    else:
        suggestion = tuner.search()
        tuner.schedule(suggestion)

    to_stop = tuner.stopping()
    if len(to_stop) > 0:
        print("==== Stopping!")
        with open(STOPPING_LOG_FILE, "a") as f:
            f.write(f"Scheduled to stop: {to_stop}\n")
        tuner.stop_runs(to_stop)


# Write custom `run_controller` function to have a little more control
# over the default one. Notably, we call our own `custom_step` function
# to avoid an issue with continuous sampling.
def run_controller(
    tuner: _WandbController,
    stop_event: threading.Event,
    sweeper_refresh_rate: float,
    verbose: bool = True,
) -> None:
    tuner._start_if_not_started()
    print("======= Sweeper Config ============")
    print(dict(tuner.sweep_config))
    print("===================================")

    while not tuner.done():
        if verbose:
            tuner.print_status()

        custom_step(tuner)

        if verbose:
            tuner.print_actions()

        if stop_event.is_set():
            wandb.termlog("Stopping wandb controller as agent is done...")
            break

        time.sleep(sweeper_refresh_rate)


def run_agent(sweep_id: str, entity: str, project: str, count: int) -> None:
    wandb.termlog("Starting wandb agent")
    wandb_agent.agent(sweep_id, entity=entity, project=project, count=count)


# This function _simulates_ setting up a sweep agent and then
# setting up an evaluating agent. It does so by spawning a thread for
# each. In reality, these would be seperate processes and likely require
# the manual step of spawning the evaluating agent with the corresponding
# sweep id that would be assigned once spinning up the sweep agent.
def sweep(
    count: int = 1,
    entity: str = ENTITY,
    project: str = PROJECT,
    sweeper_refresh_rate: float = 0.5,
):
    api = _get_client_api()
    assert api is not None

    if not api.is_authenticated:
        raise Exception("Login to W&B to use the sweep feature")

    # This is what sweeps does internally, we just use our dict version
    # to have one less file for the prototype
    #
    # > config = sweep_utils.load_sweep_config(config_yaml)
    #
    config = SWEEP_CONFIG
    wandb.termlog(f"Creating sweep from: {config}")

    # I seem to remember manually creating the `SweepConfig` object
    # as letting the controller create it from the dict version of the
    # sweep config created issues with using a 'method': 'custom'
    sweep_config = sweeps.SweepConfig(config)
    wandb.termlog(f"Loaded config: {dict(sweep_config)}")

    tuner = wandb_controller(
        sweep_id_or_config=sweep_config,
        entity=entity,
        project=project,
    )
    tuner.configure_search(my_next_run)  # type: ignore

    sweep_id = tuner.sweep_id
    wandb.termlog(f"sweep with ID: {sweep_id}")

    # DEBUG - do we get a URL from W&B?
    sweep_url = wandb_sdk.wandb_sweep._get_sweep_url(api, sweep_id)
    if sweep_url:
        wandb.termlog(f"View sweep at: {sweep_url}")

    wandb.termlog("Starting wandb controller...")

    # Create thread to run tuner in for this demo
    stop_controller = threading.Event()
    controller_thread = threading.Thread(
        target=run_controller,
        args=(tuner, stop_controller, sweeper_refresh_rate, True),
    )

    # Just to give the controller time to boot...
    wandb.termlog("Sleeping a bit to let controller start...")
    time.sleep(5)
    wandb.termlog("Starting agent")

    # Create an agent thread for this demo
    agent_thread = threading.Thread(
        target=run_agent,
        args=(sweep_id, entity, project, count),
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
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--project", type=str, default=PROJECT)

    # NOTE(eddiebergman): It seems there's quite some delay in:
    # 1. The sweep agent getting the most up to date history and issuing a stop command
    # 2. The worker agent recieving the stop command and cancelling the run.
    # ----
    # By setting the trainer-epoch-sleep-duration gives more time for this synchornization to occur
    # but it makes the entire testing process terminally slow...
    parser.add_argument("--include-early-stopping", action="store_true")
    parser.add_argument("--sweeper-refresh-rate", type=float, default=5)
    parser.add_argument("--trainer-epoch-sleep-duration", type=float, default=5)
    args = parser.parse_args()

    config = SWEEP_CONFIG.copy()
    if args.include_early_stopping:
        # Inject in a sleep into the training function to give early-stopping time to kick-in
        # This goes in conjunction with `sweeper_refresh_rate`
        # TODO(eddiebergman): I'm not sure how to increase the worker poll rate
        # so we just make the training loop slower with this
        config["parameters"]["epoch_sleep_duration"] = {
            "value": args.trainer_epoch_sleep_duration
        }
    else:
        config.pop("early_terminate", None)
        config["parameters"]["epoch_sleep_duration"] = {"value": 0.1}

    sweep(
        count=args.count,
        entity=args.entity,
        project=args.project,
        sweeper_refresh_rate=args.sweeper_refresh_rate,
    )
