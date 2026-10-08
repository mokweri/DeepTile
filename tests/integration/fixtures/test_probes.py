"""Check spatial probe layout, scalar arithmetic, and default reproducibility."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from generate_epoch_probe import generate


class ProbeTests(unittest.TestCase):
    def test_spatial_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "probe"
            generate(root, cin=3, cout=5, height=2, width=3)
            graph = json.loads((root / "graph.json").read_text())
            tensors = graph["tensors"]

            def signed(name):
                return [value if value < 128 else value - 256
                        for value in (root / tensors[name]["file"]).read_bytes()]

            inputs, weights, bias, reference = map(
                signed, ("input", "weights", "bias", "output_ref"))
            self.assertEqual(tensors["input"]["shape"], [3, 2, 3])
            self.assertEqual(tensors["output_ref"]["shape"], [5, 2, 3])
            self.assertEqual(len(reference), 30)
            for oc in range(5):
                for h in range(2):
                    for w in range(3):
                        total = bias[oc] << 4
                        for ic in range(3):
                            total += inputs[(ic * 2 + h) * 3 + w] * weights[oc * 3 + ic]
                        expected = min(127, max(-128, (total + 8) >> 4))
                        self.assertEqual(reference[(oc * 2 + h) * 3 + w], expected)

    def test_existing_default_fixture_is_unchanged(self):
        expected = {
            "input.int8.bin": "d5bded53dab8265c8d54d17bd085aceced68f9989e5cd9c0a2f8aeffb089e6a1",
            "weights.int8.bin": "232877e45a30f0d0983c1ed913296d0377d6d1fe64a38ec062540f6641ce3677",
            "bias.int8.bin": "d4bfe6fae9e6d0b543b16fbb5beb893809e5b89efd724abeb5193d864babcc07",
            "output_ref.int8.bin": "002fe68a4d57c229baf8b80fa0d4b4d8c35c481e8431cccd270b4a5a94ac9d4e",
            "graph.json": "b87db056b73460e8e86da5e1d249cf8cc637392ee98eb15038c5ca91e86ec9e1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "probe"
            generate(root)
            for name, digest in expected.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)

    def test_invalid_shape_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "probe"
            with self.assertRaises(ValueError):
                generate(root, height=0)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
