from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agent.workflow import ResearchWorkflow
from fwarl.fwa_rl import N_FEATURES, TFWA_RL
from fwarl.rl.actions import ACTIONS, POLICY_VERSION, apply_action
from fwarl.rl.learner import LinearQAgent
from fwarl.tfwa import SafeEvaluator, TFWA
from mpopt.tools.objective import ObjFunction


def evaluator(budget=400):
    obj = ObjFunction(lambda x: float(np.sum(x * x)), dim=3,
                      lb=np.full(3, -100.0), ub=np.full(3, 100.0))
    return SafeEvaluator(obj, max_eval=budget, traj_mod=0)


def agent_for(action):
    agent = LinearQAgent(N_FEATURES, len(ACTIONS), policy_version=POLICY_VERSION)
    agent.w[action, 0] = 1.0
    return agent


class DfControlTests(unittest.TestCase):
    def test_each_action_locks_native_adapt(self):
        for action in ACTIONS:
            with self.subTest(action=action["name"]):
                algo = TFWA_RL(agent=agent_for(1))
                e = evaluator()
                algo.seed = 17
                algo.init(e)
                apply_action(algo.pops, algo.init_df, action)
                commanded = [p.dist.df for p in algo.pops]
                for p in algo.pops:
                    p.evolve(e)
                self.assertEqual(commanded, [p.dist.df for p in algo.pops])
                algo.restart(0, e)
                self.assertEqual(commanded[0], algo.pops[0].dist.df)

    def test_hold_keeps_df_while_native_increases(self):
        native, rl = TFWA(), TFWA_RL(agent=agent_for(1))
        for algo in (native, rl):
            algo.seed = 9
            algo.use_restart = False
            algo.optimize(evaluator())
        self.assertTrue(any(df > native.init_df for h in native.df_his for df in h))
        self.assertTrue(all(df == rl.init_df for h in rl.df_his for df in h))
        self.assertTrue(all(t["commanded"] == t["after"] for t in rl.df_trajectory))

    def test_bounds(self):
        pop = SimpleNamespace(dist=SimpleNamespace(df=2**62))
        apply_action([pop], 5, ACTIONS[0])
        self.assertEqual(pop.dist.df, 2**62)
        pop.dist.df = 2
        apply_action([pop], 5, ACTIONS[3])
        self.assertEqual(pop.dist.df, 2)

    def test_training_and_frozen_weights(self):
        algo = TFWA_RL(train=True)
        algo.seed = 17
        algo.optimize(evaluator(1000))
        self.assertGreater(algo.agent._step, 0)
        self.assertGreater(np.linalg.norm(algo.agent.w), 0)
        self.assertTrue(algo.trajectory[-1][-1])
        for previous, following in zip(algo.trajectory, algo.trajectory[1:]):
            np.testing.assert_array_equal(previous[3], following[0])
        frozen = TFWA_RL(agent=algo.agent)
        frozen.seed = 17
        before = frozen.agent.w.copy()
        frozen.optimize(evaluator())
        np.testing.assert_array_equal(before, frozen.agent.w)

    def test_no_policy_and_old_policy_rejected(self):
        with self.assertRaises(ValueError):
            TFWA_RL()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "policy.json"
            LinearQAgent(N_FEATURES, len(ACTIONS)).save(p)
            with self.assertRaises(ValueError):
                TFWA_RL(policy_path=str(p))
            agent_for(1).save(p)
            self.assertEqual(TFWA_RL(policy_path=str(p)).agent.policy_version, POLICY_VERSION)

    def test_reward_has_no_spurious_improvement(self):
        algo = TFWA_RL(train=True)
        self.assertEqual(algo._rl_reward(12.0, 12.0), 0.0)
        self.assertGreater(algo._rl_reward(-2.0, -3.0), 0.0)
        e = evaluator()
        algo.seed = 7
        algo.init(e)
        algo._ema_improve = 0.7
        algo._rl_features(e)
        self.assertEqual(algo._ema_improve, 0.7)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.d = self.root / "iterations" / "iter_002_test"
        self.d.mkdir(parents=True)
        (self.root / "previous.json").write_text("{}")
        cfg = {"prior_artifacts": ["previous.json"],
               "train": {"benchmarks": ["cec2013"], "functions": [6], "dims": [10], "runs": 1},
               "evaluate": {"benchmarks": ["cec2022"], "functions": [9], "dims": [10], "runs": 30}}
        (self.d / "experiment.json").write_text(json.dumps(cfg))
        (self.d / "IDEA.md").write_text("假设 SETUP " + "可证伪证据及实验设计" * 20)
        (self.d / "CODE.md").write_text("控制权与奖励反思" * 30)
        self.w = ResearchWorkflow(self.root)
        self.w.start("iterations/iter_002_test")

    def tearDown(self):
        self.tmp.cleanup()

    def reviews(self):
        self.w.review("iterations/iter_002_test/IDEA.md", "根据历史结果提出可证伪假设，控制单一变量，独立训练测试并验证预算与种子。")
        self.w.review("iterations/iter_002_test/CODE.md", "原生 adapt 仍会增长自由度，锁定策略指令并测试重启，必须重新训练冻结评测。")

    def test_early_finish_and_stage_skip_rejected(self):
        self.assertIsNotNone(self.w.finish_error())
        with self.assertRaises(ValueError):
            self.w.run("freeze")
        self.reviews()
        self.assertEqual(self.w.state["stage"], "smoke")
        self.assertIsNotNone(self.w.finish_error())

    def test_failure_remains_and_resumes(self):
        self.reviews()
        with patch.object(self.w, "_command", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                self.w.run("smoke")
        restored = ResearchWorkflow(self.root)
        restored.start("iterations/iter_002_test")
        self.assertEqual(restored.state["stage"], "smoke")

    def test_new_code_invalidates_review(self):
        self.reviews()
        (self.root / "fwarl").mkdir()
        (self.root / "fwarl" / "new.py").write_text("x = 1\n")
        with self.assertRaises(ValueError):
            self.w.run("smoke")

    def test_old_completed_experiment_cannot_finish_new_session(self):
        self.reviews()
        self.w.state["stage"] = "completed"
        self.w.state["events"].append({"kind": "verified_experiment", "phase": "freeze", "time": 1.0})
        self.assertIsNotNone(self.w.finish_error(after=time.time()))

    def test_shell_failure_is_recorded(self):
        with self.assertRaises(RuntimeError):
            self.w._command("failure", ["python3", "-c", "raise SystemExit(7)"])
        self.assertEqual(self.w.state["events"][-1]["returncode"], 7)
        self.assertTrue((self.root / self.w.state["events"][-1]["log"]).exists())

    def test_react_rejects_model_finish_without_experiment(self):
        from agent import run_agent
        from agent.tools import ToolBox
        client = SimpleNamespace(last_usage={}, chat=lambda *a, **kw:
                                 '{"tool":"finish","args":{"summary":"立项完成"}}')
        sess = self.root / "session.md"
        with patch.object(run_agent, "log_token"):
            stats = run_agent.run_session(client, ToolBox({}), "test", "df control", sess,
                                          1, {"mode": "goal"}, lambda *a: None)
        self.assertFalse(stats["finished"])
        self.assertEqual(stats["exit_reason"], "max_turns")
        self.assertIn("finish 被拒绝", sess.read_text())

    def test_worker_receives_explicit_runs(self):
        from experiments import run_eval
        task = ("tfwa", "cec2022", 9, 10, 0, 42, 2000, False, None, False, 2)
        with patch.object(run_eval, "run_single", return_value=({"status": "success"}, None)) as solve:
            self.assertEqual(len(run_eval._worker(task)), 2)
            self.assertEqual(solve.call_count, 2)


if __name__ == "__main__":
    unittest.main()
