from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from self_judged_refusal_direction.errors import RefusalDirectionError

COMMANDS = (
    "inspect-model",
    "generate-baseline-trajectories",
    "judge-baseline-trajectories",
    "collect-activations",
    "build-candidates",
    "evaluate-candidates",
    "export-model",
    "evaluate-export",
    "run",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
    return parser


def command_functions() -> dict[str, Callable[[str], None]]:
    from self_judged_refusal_direction import pipeline

    return {
        "inspect-model": pipeline.inspect_model,
        "generate-baseline-trajectories": pipeline.generate_baseline_trajectories,
        "judge-baseline-trajectories": pipeline.judge_baseline_trajectories,
        "collect-activations": pipeline.collect_activations,
        "build-candidates": pipeline.build_direction_candidates,
        "evaluate-candidates": pipeline.evaluate_candidates,
        "export-model": pipeline.export_model,
        "evaluate-export": pipeline.evaluate_export,
        "run": pipeline.run_pipeline,
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        command_functions()[args.command](args.config)
    except RefusalDirectionError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
