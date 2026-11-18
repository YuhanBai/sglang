"""Exercise Ascend 950's packed LongCat history update and OE hash."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="nightly-acc-test-4-npu-a5", nightly=True)


class TestLongCatOeA5(unittest.TestCase):
    def test_packed_history_and_hash(self):
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        if not is_npu_arch35():
            self.skipTest("Ascend 950 required")

        from sglang.srt.hardware_backend.npu.longcat_oe import (
            compute_ngram_ids,
            update_token_table,
        )

        def device_tensor(values):
            return torch.tensor(values, device="npu", dtype=torch.int32)

        history = torch.full((3, 12), -1, device="npu", dtype=torch.int32)
        lengths = device_tensor([4, 0, 3])
        rows = device_tensor([2, 1, 0]).long()
        starts = device_tensor([0, 0, 4])
        tokens = device_tensor([1, 2, 5, 2, 3, 4, 7])
        update_token_table(tokens, history, rows, starts, lengths)
        expected_history = [
            [-1] * 4 + [3, 4, 7] + [-1] * 5,
            [-1] * 12,
            [1, 2, 5, 2] + [-1] * 8,
        ]
        self.assertEqual(history.cpu().tolist(), expected_history)

        n, splits, eos = 5, 4, 2
        mods = [31 + 2 * table for table in range((n - 1) * splits)]
        factors = [[pow(17, d, mod) for d in range(n)] for mod in mods]
        offsets = [sum(mods[:table]) for table in range(len(mods))]
        # EOS is included when current, but cuts the history when encountered
        # during lookback. The final row represents graph padding.
        requests = [2, 2, 2, 2, 0, 0, 0, 1]
        positions = [0, 1, 2, 3, 4, 5, 6, 0]
        valid = [True] * 7 + [False]
        result = torch.empty(
            (len(positions), len(mods)), device="npu", dtype=torch.int32
        )
        compute_ngram_ids(
            history,
            device_tensor(positions),
            device_tensor(requests).long(),
            device_tensor(valid).bool(),
            device_tensor(factors).view(n - 1, splits, n),
            device_tensor(mods).view(n - 1, splits),
            device_tensor(offsets),
            eos,
            result,
        )
        expected = []
        for row, position, active in zip(requests, positions, valid, strict=True):
            ids = []
            for table, mod in enumerate(mods):
                total = 0
                if active:
                    for distance in range(min(position + 1, table // splits + 2)):
                        token = expected_history[row][position - distance]
                        if token < 0 or (distance > 0 and token == eos):
                            break
                        total += token * factors[table][distance]
                ids.append(total % mod + offsets[table])
            expected.append(ids)
        self.assertEqual(result.cpu().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
