"""Regression test for checkpoint architecture dispatch."""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.hf_transformers.config import HfModelConfigParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLongCat25Config(unittest.TestCase):
    def test_native_dispatch_keeps_the_normalized_model_type(self):
        cases = [
            ("flash", {}, "LongcatFlashForCausalLM"),
            (
                "pro_v",
                {
                    "visual_config": {"enable": True, "image_pad_token_id": 221},
                    "visual_quantizer_config": {"enable": True},
                },
                "Longcat25ForConditionalGeneration",
            ),
        ]
        for raw_model_type, extra, architecture in cases:
            with (
                self.subTest(raw_model_type=raw_model_type),
                tempfile.TemporaryDirectory() as folder,
            ):
                data = {
                    "model_type": raw_model_type,
                    "architectures": ["LongcatCausalLM"],
                    **extra,
                }
                (Path(folder) / "config.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )
                config = HfModelConfigParser().parse(folder, trust_remote_code=False)
                self.assertEqual(config.model_type, "longcat_flash")
                self.assertEqual(config.architectures, [architecture])
                if extra:
                    self.assertEqual(config.image_token_id, 221)


if __name__ == "__main__":
    unittest.main()
