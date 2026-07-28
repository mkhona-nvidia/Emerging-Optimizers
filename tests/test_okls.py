# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from absl import flags, logging
from absl.testing import absltest, parameterized

from emerging_optimizers.soap.okls import OKLS


flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to run tests on")
flags.DEFINE_integer("seed", None, "Random seed for reproducible tests")
FLAGS = flags.FLAGS


def setUpModule() -> None:
    if FLAGS.seed is not None:
        logging.info("Setting random seed to %d", FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FLAGS.seed)


class OKLSTest(parameterized.TestCase):
    @parameterized.parameters((4, 3), (2, 4, 3))  # type: ignore[misc]
    def test_step_initializes_state_and_updates_parameter(self, shape: tuple[int, ...]) -> None:
        param = torch.nn.Parameter(torch.randn(*shape, device=FLAGS.device))
        original = param.detach().clone()
        param.grad = torch.randn_like(param)
        optimizer = OKLS([param], lr=0.01, ridge_eps=1e-9)

        optimizer.step()

        self.assertFalse(torch.equal(param, original))
        self.assertTrue(torch.isfinite(param).all())
        state = optimizer.state[param]
        self.assertEqual(state["step"], 1)
        self.assertCountEqual(state.keys(), ["step", "momentum", "S_a", "S_b", "P_a", "P_b"])
        for value in state.values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())

    def test_rejects_non_matrix_parameter(self) -> None:
        param = torch.nn.Parameter(torch.randn(4, device=FLAGS.device))
        with self.assertRaisesRegex(ValueError, "only supports 2D or 3D"):
            OKLS([param], lr=0.01)


if __name__ == "__main__":
    absltest.main()
