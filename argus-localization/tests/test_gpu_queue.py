"""GPU selection for scripts/gpu_queue.py, from canned nvidia-smi output."""

import unittest

from scripts.gpu_queue import busy_reasons, parse_apps, parse_gpus, pick_idle_gpu

GPUS_CSV = """\
0, GPU-aaaa, NVIDIA RTX A6000, 38000, 49140, 97
1, GPU-bbbb, NVIDIA RTX A6000, 4, 49140, 0
"""
APPS_CSV = """\
GPU-aaaa, 4242, python, 37990
"""


class GpuQueueTest(unittest.TestCase):
    def setUp(self):
        self.gpus = parse_gpus(GPUS_CSV)
        self.apps = parse_apps(APPS_CSV)

    def test_parsing(self):
        self.assertEqual([g.index for g in self.gpus], [0, 1])
        self.assertEqual(self.gpus[0].memory_used_mb, 38000)
        self.assertEqual(self.apps[0].pid, 4242)
        self.assertEqual(parse_gpus(""), [])
        self.assertEqual(parse_gpus("0, GPU-x, T4, [N/A], 15360, [N/A]\n")[0].utilization_pct, 0)

    def test_busy_gpu_is_skipped(self):
        self.assertIn("pid 4242", busy_reasons(self.gpus[0], self.apps, 1024, 10)[0])
        self.assertEqual(pick_idle_gpu(self.gpus, self.apps, None, 1024, 10).index, 1)

    def test_nothing_idle_means_wait(self):
        apps = parse_apps(APPS_CSV + "GPU-bbbb, 777, trainer, 100\n")
        self.assertIsNone(pick_idle_gpu(self.gpus, apps, None, 1024, 10))

    def test_memory_or_utilization_alone_mark_busy(self):
        # No listed process (e.g. another container), but memory is in use.
        gpus = parse_gpus("0, GPU-aaaa, A6000, 20000, 49140, 0\n")
        self.assertIsNone(pick_idle_gpu(gpus, [], None, 1024, 10))
        gpus = parse_gpus("0, GPU-aaaa, A6000, 10, 49140, 55\n")
        self.assertIsNone(pick_idle_gpu(gpus, [], None, 1024, 10))

    def test_allowed_and_require_all_idle(self):
        self.assertIsNone(pick_idle_gpu(self.gpus, self.apps, {0}, 1024, 10))
        self.assertIsNone(pick_idle_gpu(self.gpus, self.apps, None, 1024, 10, require_all_idle=True))
        self.assertEqual(pick_idle_gpu(self.gpus, [], {1}, 100000, 100, require_all_idle=True).index, 1)


if __name__ == "__main__":
    unittest.main()
