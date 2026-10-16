"""Prepare an immutable candidate; this does not qualify or start a server."""
import argparse
import json
from pathlib import Path
from inferx.ebplus_validation import prepare_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-scheduler', required=True, type=Path)
    parser.add_argument('--author-calibration', required=True, type=Path)
    parser.add_argument('--native-scheduler', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_adapter(args.author_scheduler, args.author_calibration,
                                    args.native_scheduler, args.output), indent=2))


if __name__ == '__main__':
    main()
