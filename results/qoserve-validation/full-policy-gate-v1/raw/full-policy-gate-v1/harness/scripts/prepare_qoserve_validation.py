"""Prepare source-bound QoServe overlays without importing a GPU runtime."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from inferx.qoserve_validation import prepare_adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author-root',type=Path,required=True)
    parser.add_argument('--native-scheduler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coefficients',type=Path,help='Two-bin coefficient JSON; presence does not qualify a profile.')
    args=parser.parse_args()
    coefficients=json.loads(args.coefficients.read_text()) if args.coefficients else None
    manifest=prepare_adapter(args.author_root,args.native_scheduler,args.output,coefficients=coefficients)
    print(json.dumps(dict(destination=str(args.output),
                         deployment_coefficients_supplied=manifest['policy']['deployment_coefficients_supplied'])))


if __name__=='__main__':main()
