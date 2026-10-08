"""Check the clamped-activation oracles reach their clamps and stay reproducible."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from generate_activation_cases import generate, relu6_max


class ActivationCaseTests(unittest.TestCase):
    def test_relu6_clamp_value(self):
        self.assertEqual(relu6_max(4), 96)
        self.assertEqual(relu6_max(3), 48)
        self.assertEqual(relu6_max(5), 127)   # 6.0 no longer fits in int8

    def test_cases_reach_their_clamps_and_are_unchanged(self):
        expected = {
            "k5_dense": "73bddf48388d5856731536711b22939c054762549dfc028adeac714fb209bfca",
            "k5_depthwise_s2": "ce0615907ef60b8c285c6e97470157171d81d23ed3c17d712ba4207cbb19ec71",
            "k5_depthwise_wide": "5da57f97466aaa3b6f9d46e2317954d6ea5df1174b3830c4e02d7c643a109135",
            "relu6_dense_k3": "4e204a90d04e8ddc212ab05bc2a9f8a5c6827298c9a48e5622a2f3f95c00da29",
            "relu6_fused_pwdw": "a22323217069f9aa973998e9b137ef12f1f440a35bbb0ab8a4e5fb03de31e92a",
            "relu6_fused_pwdw5": "f4e0d004e44b149899b5ba3e7a8f06ce7269c838a5f4cbffa149c233fce63815",
            "relu6_post_add": "067597d22b3f5307860eebcdb9953860521aeb47ff92cbacfac21db641b21d6f",
        }
        with tempfile.TemporaryDirectory() as directory:
            for graph_path in generate(Path(directory) / "cases"):
                graph = json.loads(graph_path.read_text())
                output = graph["graph"]["outputs"][0]
                reference = graph["tensors"][graph["graph"]["references"][output]]
                data = (graph_path.parent / reference["file"]).read_bytes()
                name = graph_path.parent.name
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected[name], name)

                values = [v if v < 128 else v - 256 for v in data]
                post = graph["nodes"][-1]["post_ops"]
                if post.get("relu6") or post.get("post_add_relu6"):
                    self.assertEqual(max(values), relu6_max(reference["frac"]), name)
                self.assertEqual(min(values), 0, name)


if __name__ == "__main__":
    unittest.main()
