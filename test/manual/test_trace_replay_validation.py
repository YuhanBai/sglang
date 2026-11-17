"""Validate trace replay against an already running server, without launching one.

Set TRACE_REPLAY_URL and TRACE_REPLAY_EVIDENCE_DIR to select the server and
retain request/response evidence. Run with the same checkout as the server.
"""

import json
import math
import os
import unittest
from pathlib import Path

import requests

from sglang.test.test_utils import CustomTestCase


class TestTraceReplayValidation(CustomTestCase):
    base_url = os.environ.get("TRACE_REPLAY_URL", "http://127.0.0.1:31087")
    evidence_dir = Path(os.environ.get("TRACE_REPLAY_EVIDENCE_DIR", "trace-evidence"))

    @classmethod
    def setUpClass(cls):
        cls.evidence_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(cls.base_url + "/get_model_info", timeout=30)
        response.raise_for_status()
        cls.model_info = response.json()
        (cls.evidence_dir / "model_info.json").write_text(
            json.dumps(cls.model_info, indent=2), encoding="utf-8"
        )

    def generate(self, label, payload):
        response = requests.post(self.base_url + "/generate", json=payload, timeout=180)
        (self.evidence_dir / f"{label}.json").write_text(
            json.dumps(
                {
                    "request": payload,
                    "status": response.status_code,
                    "response": response.json(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def payload(self, trace, **sampling):
        return {
            "input_ids": [9707, 11, 1070],
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": 64,
                "trace_decode_token_ids": trace,
                **sampling,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
        }

    def assert_trace(self, result, trace):
        self.assertEqual(result["output_ids"], trace)
        meta = result["meta_info"]
        self.assertEqual(meta["completion_tokens"], len(trace))
        entries = meta["output_token_logprobs"]
        self.assertEqual([entry[1] for entry in entries], trace)
        self.assertTrue(
            all(entry[0] is not None and math.isfinite(entry[0]) for entry in entries)
        )

    def test_replay_logprobs_match_across_temperatures(self):
        """Replay must preserve the model's token scores across temperatures."""
        trace = [1, 2, 3, 4, 5, 6, 7, 8]
        replay = self.generate("logprob_replay", self.payload(trace, temperature=1.0))
        greedy = self.generate("logprob_greedy", self.payload(trace, temperature=0.0))
        self.assert_trace(replay, trace)
        self.assert_trace(greedy, trace)
        replay_scores = replay["meta_info"]["output_token_logprobs"]
        for replay_entry, greedy_entry in zip(
            replay_scores, greedy["meta_info"]["output_token_logprobs"], strict=True
        ):
            self.assertAlmostEqual(replay_entry[0], greedy_entry[0], delta=1e-5)
        if os.environ.get("TRACE_REPLAY_EXPECT_SPECULATIVE") == "1":
            self.assertGreater(replay["meta_info"]["spec_verify_ct"], 0)

    def test_stop_conditions_and_cap(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_info["model_path"], local_files_only=True
        )
        trace = [1, tokenizer.eos_token_id, 2, 3]
        result = self.generate(
            "stop_ignored", self.payload(trace, stop_token_ids=[1], stop=["!"])
        )
        self.assert_trace(result, trace)
        result = self.generate("max_tokens_cap", self.payload(trace, max_new_tokens=2))
        self.assert_trace(result, trace[:2])

    def test_mixed_batch(self):
        traces = [[1, 2, 3], None, [4, 5, 6, 7, 8, 9, 10]]
        result = self.generate(
            "mixed_batch",
            {
                "text": [
                    "A short prompt",
                    "The capital of France is",
                    "A longer replay prompt",
                ],
                "sampling_params": [
                    {
                        "temperature": 0,
                        "max_new_tokens": 12,
                        "trace_decode_token_ids": trace,
                    }
                    for trace in traces
                ],
                "return_logprob": True,
            },
        )
        self.assert_trace(result[0], traces[0])
        self.assert_trace(result[2], traces[2])
        self.assertGreater(result[1]["meta_info"]["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
