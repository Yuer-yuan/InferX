"""CPU regression for the installed SGLang sampler; run in its own venv.

The singleton fixture rejects any collective (standing in for lazy NCCL
allocation); two real Gloo ranks verify that token-wise MIN still works.
No CUDA device, model load, or project pytest dependency is required.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import multiprocessing
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from sglang.srt.layers import sampler


def reduce_tokens(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        tokens = torch.tensor([7, 11] if rank == 0 else [3, 15])
        owner = SimpleNamespace(tp_sync_group=dist.group.WORLD)
        sampler.Sampler._sync_token_ids_across_tp(
            owner, tokens, SimpleNamespace(grammars=[object()])
        )
        output.put((rank, tokens.tolist()))
    finally:
        dist.destroy_process_group()


class SamplerSyncRegression(unittest.TestCase):
    def test_singleton_grammar_never_initializes_collective(self):
        with TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo", init_method=Path(directory, "rank").as_uri(),
                rank=0, world_size=1,
            )
            try:
                tokens = torch.tensor([7, 11])
                owner = SimpleNamespace(tp_sync_group=dist.group.WORLD)
                with patch.object(
                    dist, "all_reduce", side_effect=RuntimeError("lazy NCCL allocation")
                ):
                    sampler.Sampler._sync_token_ids_across_tp(
                        owner, tokens, SimpleNamespace(grammars=[object()])
                    )
                self.assertEqual(tokens.tolist(), [7, 11])
            finally:
                dist.destroy_process_group()

    def test_two_ranks_still_compute_tokenwise_min(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        with TemporaryDirectory() as directory:
            rendezvous = Path(directory, "ranks").as_uri()
            workers = [context.Process(target=reduce_tokens,
                       args=(rank, rendezvous, output)) for rank in range(2)]
            try:
                for worker in workers:
                    worker.start()
                results = [output.get(timeout=90) for _ in workers]
                for worker in workers:
                    worker.join(timeout=30)
                    self.assertEqual(worker.exitcode, 0)
                self.assertEqual(sorted(results), [(0, [3, 11]), (1, [3, 11])])
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                output.close()


if __name__ == "__main__":
    unittest.main()
