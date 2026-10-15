"""Prepare a standalone, hashed Python-source overlay on the server HDD."""
import argparse
import json
import os
from pathlib import Path

from inferx.sglang_pd import prepare_overlay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--policy',choices=['control','reserve','responsive','pdmux','pdmux-chunked'],required=True)
    parser.add_argument('--cap',type=int)
    parser.add_argument('--protect-decode',action='store_true')
    parser.add_argument('--credit-owned-slots',action='store_true')
    parser.add_argument('--bind-green-contexts',action='store_true')
    args=parser.parse_args()
    root=Path(os.environ['INFERX_ROOT']).resolve()
    if root!=Path('/home/guokc/mypro/gnn-mount-sda1/InferX'):
        raise RuntimeError('server HDD root required')
    if not args.name or Path(args.name).name!=args.name or args.name in {'.','..'}:
        parser.error('simple new overlay name required')
    source=root/'pd-baselines/sglang0519-venv/lib/python3.12/site-packages/sglang'
    destination=root/'pd-strategies/overlays'/args.name
    print(json.dumps(prepare_overlay(source,destination,args.policy,args.cap,args.protect_decode,
                                    args.credit_owned_slots,args.bind_green_contexts),indent=2))


if __name__=='__main__':main()
