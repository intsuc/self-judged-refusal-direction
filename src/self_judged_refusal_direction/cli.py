from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import cast

from self_judged_refusal_direction.errors import RefusalDirectionError

COMMAND_FUNCTION_NAMES = {
    "inspect-model": "inspect_model",
    "validate-judge": "validate_judge",
    "generate-baseline-trajectories": "generate_baseline_trajectories",
    "judge-baseline-trajectories": "judge_baseline_trajectories",
    "collect-activations": "collect_activations",
    "build-candidates": "build_direction_candidates",
    "evaluate-candidates": "evaluate_candidates",
    "export-model": "export_model",
    "evaluate-export": "evaluate_export",
    "run": "run_pipeline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMAND_FUNCTION_NAMES:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
    return parser


def command_function(command: str) -> Callable[[str], None]:
    from self_judged_refusal_direction import pipeline

    return cast(Callable[[str], None], getattr(pipeline, COMMAND_FUNCTION_NAMES[command]))


def main() -> None:
    args = build_parser().parse_args()
    try:
        command_function(args.command)(args.config)
    except RefusalDirectionError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
