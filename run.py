from __future__ import annotations

import argparse
import sys
import time

from stages import code, collect, deploy, label, report, validate
from stages.common import append_log, print_cost_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the medical benchmark disease index.")
    parser.add_argument("--stage", choices=["all", "collect", "label", "code", "validate", "report", "deploy"], default="all")
    parser.add_argument("--approve-pilot", action="store_true", help="Continue beyond the required 50-item pilot after Dominic says go.")
    parser.add_argument("--allow-paid", action="store_true", help="Allow paid Gemini only after Dominic explicitly approves and GEMINI_PAID=1 is set.")
    parser.add_argument("--skip-model-pull", action="store_true", help="Do not automatically download missing free Ollama models.")
    parser.add_argument("--deploy", action="store_true", help="Publish the redacted explorer as a public Hugging Face Space. Off unless asked, because it is public.")
    parser.add_argument("--limit", type=int, default=None, help="Collect only N total items for a recorded smoke test; never use for final paper counts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    allow_pull = not args.skip_model_pull
    try:
        if args.stage in {"all", "collect"}:
            collect.run(limit=args.limit)
        if args.stage in {"all", "label"}:
            finished = label.run(approve_pilot=args.approve_pilot, allow_paid=args.allow_paid, allow_pull=allow_pull)
            if not finished:
                print_cost_report(started)
                return 2
        if args.stage in {"all", "code"}:
            code.run()
        if args.stage in {"all", "validate"}:
            validate.run(allow_paid=args.allow_paid, allow_pull=allow_pull)
        if args.stage in {"all", "report"}:
            report.run()
        if args.stage == "deploy" or (args.stage == "all" and args.deploy):
            deploy.run()
        print_cost_report(started)
        return 0
    except Exception as exc:
        append_log("Run failed", f"Stage: `{args.stage}`  \nError: `{type(exc).__name__}: {exc}`")
        print_cost_report(started)
        print(f"Pipeline stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
