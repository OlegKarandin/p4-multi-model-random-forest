"""python -m src.verify: --run DIR [--workers N] | --archive DIR [--workers N] |
--model-archive DIR |
--rescore --run DIR (spec 2026-09-29 §6.6)."""
import argparse
import sys

from src.verify import model_archive, runner


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m src.verify')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', metavar='DIR', help='campaign run directory')
    source.add_argument('--archive', metavar='DIR',
                        help='calibration archive: RECOMPILE every design and compare p4c with '
                             'the archived p4c logs (tests the environment, not the model)')
    source.add_argument('--model-archive', metavar='DIR',
                        help='calibration archive: replay the MODEL against the archived '
                             'p4c logs; exit 1 on any miss not in model_archive.KNOWN_MISSES')
    parser.add_argument('--rescore', action='store_true',
                        help='recompute the model side of --run; no compiles')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)

    if args.rescore:
        if not args.run:
            parser.error('--rescore needs --run DIR')
        runner.rescore(args.run)
        return 0
    if args.model_archive:
        return model_archive.main_exit_code(model_archive.check(args.model_archive))
    if args.archive:
        results = runner.verify_archive(args.archive, workers=args.workers)
        return 0 if all(r['match'] for r in results) else 1
    return runner.run(args.run, workers=args.workers)


if __name__ == '__main__':
    sys.exit(main())
