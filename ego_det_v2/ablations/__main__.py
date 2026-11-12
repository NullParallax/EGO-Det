from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import register_full, summarize, train
from .registry import ABLATION_TABLES, ABLATIONS


def _list_variants(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ego_det_v2.ablations list"
    )
    parser.add_argument("--keys-only", action="store_true")
    parser.add_argument(
        "--without-full",
        action="store_true",
        help="omit the completed full control and list only variants to train",
    )
    args = parser.parse_args(argv)
    variants = [
        key for key in ABLATIONS if not args.without_full or key != "full"
    ]
    if args.keys_only:
        for key in variants:
            print(key)
        return
    table_memberships = {
        key: [
            table_key
            for table_key, table in ABLATION_TABLES.items()
            if any(row.variant == key for row in table.rows)
        ]
        for key in ABLATIONS
    }
    payload = [
        {
            "variant": key,
            "label": spec.label,
            "description": spec.description,
            "overrides": spec.override_dict(),
            "tables": table_memberships[key],
        }
        for key, spec in ABLATIONS.items()
        if key in variants
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: python -m ego_det_v2.ablations "
            "{list,register-full,train,summarize} ...\n\n"
            "commands:\n"
            "  list       show canonical variants and table reuse\n"
            "  register-full  reuse the completed full run as control\n"
            "  train      train one independently initialized variant\n"
            "  summarize  write controlled JSON/CSV result tables"
        )
        return
    command, rest = arguments[0], arguments[1:]
    if command == "list":
        _list_variants(rest)
    elif command == "register-full":
        register_full.main(rest)
    elif command == "train":
        train.main(rest)
    elif command == "summarize":
        summarize.main(rest)
    else:
        raise SystemExit(
            f"unknown command {command!r}; choose list, register-full, "
            "train, or summarize"
        )


if __name__ == "__main__":
    main()
