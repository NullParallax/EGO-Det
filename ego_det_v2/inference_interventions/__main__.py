from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import evaluate, summarize
from .registry import INTERVENTIONS, TABLES


def _list_interventions(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ego_det_v2.inference_interventions list"
    )
    parser.add_argument("--keys-only", action="store_true")
    args = parser.parse_args(argv)
    if args.keys_only:
        for key in INTERVENTIONS:
            print(key)
        return
    memberships = {
        key: [
            table_key
            for table_key, table in TABLES.items()
            if any(row.variant == key for row in table.rows)
        ]
        for key in INTERVENTIONS
    }
    payload = [
        {
            "variant": key,
            "label": spec.label,
            "description": spec.description,
            "config_variant": spec.config_variant,
            "config_overrides": spec.config_overrides(),
            "runtime_transform": spec.runtime_transform,
            "tables": memberships[key],
        }
        for key, spec in INTERVENTIONS.items()
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: python -m ego_det_v2.inference_interventions "
            "{list,evaluate,summarize} ...\n\n"
            "commands:\n"
            "  list       show inference-only interventions\n"
            "  evaluate   evaluate one intervention on COCO val2017\n"
            "  summarize  aggregate metrics into separate CSV/JSON tables"
        )
        return
    command, rest = arguments[0], arguments[1:]
    if command == "list":
        _list_interventions(rest)
    elif command == "evaluate":
        evaluate.main(rest)
    elif command == "summarize":
        summarize.main(rest)
    else:
        raise SystemExit(
            f"unknown command {command!r}; choose list, evaluate, or summarize"
        )


if __name__ == "__main__":
    main()
