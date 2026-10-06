"""One entry point for every experiment in the paper.

    python -m mgqc list
    python -m mgqc show accuracy
    python -m mgqc run accuracy --configs A1,B5 --seeds 20 --jobs 26
    python -m mgqc selftest

`run` passes everything after the experiment name straight to that experiment's
own parser, so the per-experiment flags are unchanged from the scripts these
modules were assembled from. `python -m mgqc run accuracy --help` prints them.
"""
import argparse
import sys

from . import __version__
from .experiments import REGISTRY, load


def cmd_list(_):
    width = max(len(k) for k in REGISTRY)
    print(f"\n  {len(REGISTRY)} experiments\n")
    for name in REGISTRY:
        experiment = REGISTRY[name]
        print(f"  {name:<{width}}  {experiment.summary}")
    print("\n  python -m mgqc show <name>   for the outputs and the "
          "manuscript items it feeds\n")
    return 0


def cmd_show(args):
    if args.name not in REGISTRY:
        print(f"  unknown experiment: {args.name}")
        return cmd_list(args) or 1
    experiment = REGISTRY[args.name]
    print(f"\n  {args.name}\n")
    print(f"    what        {experiment.summary}")
    print(f"    writes      {experiment.produces}")
    print(f"    feeds       {experiment.manuscript}")
    print(f"    code        mgqc.experiments.{experiment.module}.{experiment.entry}")
    print(f"\n  python -m mgqc run {args.name} --help\n")
    return 0


def cmd_run(args):
    if args.name not in REGISTRY:
        print(f"  unknown experiment: {args.name}\n")
        return cmd_list(args) or 1
    return_code = load(args.name)(args.rest)
    return 0 if return_code is None else return_code


def cmd_selftest(args):
    from . import selftest
    return selftest.main(args.rest) if hasattr(selftest, "main") else _run_selftest()


def _run_selftest():
    from . import selftest
    import runpy
    runpy.run_module("mgqc.selftest", run_name="__main__")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mgqc",
        description="Bond-type-conditioned coupling in graph-structured "
                    "quantum circuits - experiment runner.")
    parser.add_argument("--version", action="version", version=f"mgqc {__version__}")
    subparsers = parser.add_subparsers(dest="cmd")

    subparser = subparsers.add_parser("list", help="every experiment and what it does")
    subparser.set_defaults(fn=cmd_list)

    subparser = subparsers.add_parser("show", help="what one experiment writes and feeds")
    subparser.add_argument("name")
    subparser.set_defaults(fn=cmd_show)

    subparser = subparsers.add_parser("run", help="run one experiment")
    subparser.add_argument("name")
    subparser.add_argument("rest", nargs=argparse.REMAINDER,
                           help="flags for that experiment; try --help")
    subparser.set_defaults(fn=cmd_run)

    subparser = subparsers.add_parser("selftest", help="gradient and measurement checks")
    subparser.add_argument("rest", nargs=argparse.REMAINDER)
    subparser.set_defaults(fn=cmd_selftest)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
