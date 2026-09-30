"""python -m src.verify: --run DIR [--workers N] | --archive DIR [--workers N] |
--rescore --run DIR (spec 2026-09-29 §6.6)."""
import argparse
import sys

from src.verify import runner


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m src.verify')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', metavar='DIR', help='campaign run directory')
    source.add_argument('--archive', metavar='DIR', help='calibration archive directory')
    parser.add_argument('--rescore', action='store_true',
                        help='recompute the model side of --run; no compiles')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)

    if args.rescore:
        if not args.run:
            parser.error('--rescore needs --run DIR')
        runner.rescore(args.run)
        return 0
    if args.archive:
        results = runner.verify_archive(args.archive, workers=args.workers)
        return 0 if all(r['match'] for r in results) else 1
    return runner.run(args.run, workers=args.workers)


if __name__ == '__main__':
    sys.exit(main())
