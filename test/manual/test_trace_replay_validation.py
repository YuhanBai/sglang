"""Validate trace replay against an already running server, without launching one.

Set TRACE_REPLAY_URL and TRACE_REPLAY_EVIDENCE_DIR to select the server and
retain request/response evidence. Run with the same checkout as the server.
"""

import json
import math
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
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

    def generate(self, label, payload, status=200):
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
        self.assertEqual(response.status_code, status, response.text)
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

    def test_basic_and_greedy(self):
        for temperature in [0, 0.8, 1.0]:
            with self.subTest(temperature=temperature):
                trace = [1, 2, 3, 4]
                result = self.generate(
                    f"basic_{temperature}", self.payload(trace, temperature=temperature)
                )
                self.assert_trace(result, trace)

    def test_logprob_matches_teacher_forcing(self):
        """Forcing a token must preserve its model probability, including decode steps."""
        trace = [1, 2, 3, 4, 5, 6]
        payload = self.payload(trace)
        replay = self.generate("logprob_replay", payload)
        self.assert_trace(replay, trace)
        score = self.generate(
            "logprob_teacher",
            {
                "input_ids": payload["input_ids"] + trace,
                "sampling_params": {"temperature": 1.0, "max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
        # Include the causal predecessor in the scoring window: its first
        # entry has no logprob, whereas every trace token must have one.
        teacher = score["meta_info"]["input_token_logprobs"][
            len(payload["input_ids"]) :
        ]
        self.assertEqual([entry[1] for entry in teacher], trace)
        errors = [
            abs(a[0] - b[0])
            for a, b in zip(replay["meta_info"]["output_token_logprobs"], teacher)
        ]
        (self.evidence_dir / "logprob_comparison.json").write_text(
            json.dumps(
                {
                    "absolute_errors": errors,
                    "max_absolute_error": max(errors),
                    "tolerance": 0.1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # BF16 prefill/decode kernels can round differently; altered or masked
        # distributions produce substantially larger differences on these tokens.
        self.assertLessEqual(max(errors), 0.1)

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

    def test_concurrent_variable_lengths(self):
        def run(index):
            trace = list(range(10 + index, 10 + index + 2 * index + 1))
            payload = self.payload(trace)
            # Distinct long prompts exercise chunked prefill and prefix reuse.
            payload["input_ids"] = [9707 + index] * (129 + index * 17)
            result = self.generate(f"concurrent_{index}", payload)
            return result, trace

        with ThreadPoolExecutor(max_workers=6) as pool:
            for result, trace in pool.map(run, range(6)):
                self.assert_trace(result, trace)

    def test_streaming(self):
        trace = [1, 2, 3, 4, 5, 6, 7]
        payload = self.payload(trace)
        payload["stream"] = True
        chunks = []
        with requests.post(
            self.base_url + "/generate", json=payload, stream=True, timeout=180
        ) as response:
            self.assertEqual(
                response.status_code,
                200,
                response.text if response.status_code != 200 else "",
            )
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    chunks.append(json.loads(line[6:]))
        (self.evidence_dir / "streaming.json").write_text(
            json.dumps(chunks, indent=2), encoding="utf-8"
        )
        self.assertTrue(chunks)
        # The default native stream returns cumulative IDs and logprobs.
        self.assert_trace(chunks[-1], trace)
        for chunk in chunks:
            ids = chunk["output_ids"]
            self.assertEqual(ids, trace[: len(ids)])

    def test_reject_invalid_requests(self):
        cases = {
            "empty": ([], {}),
            "negative": ([-1], {}),
            "oov": ([999999999], {}),
            "grammar": ([1], {"regex": "[a-z]+"}),
            "parallel_sampling": ([1], {"n": 2}),
        }
        for label, (trace, sampling) in cases.items():
            with self.subTest(label=label):
                self.generate(
                    f"reject_{label}", self.payload(trace, **sampling), status=400
                )


if __name__ == "__main__":
    unittest.main()
