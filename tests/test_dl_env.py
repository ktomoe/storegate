"""Tests for storegate.task.dl_env."""

import pytest

from storegate.task.dl_env import (
    DLEnv,
    DataLoader,
    DataSet,
    LearningMode,
    Metrics,
    Phase,
    VarName,
    _PhaseLists,
    _PhaseSlots,
)
from storegate.task.pytorch_metrics import ResolvedMetric


# ---------------------------------------------------------------------------
# LearningMode
# ---------------------------------------------------------------------------
class TestLearningMode:
    def test_values(self):
        assert LearningMode.SUPERVISED == "supervised"
        assert LearningMode.SELF_SUPERVISED == "self_supervised"
        assert LearningMode.UNSUPERVISED == "unsupervised"

    def test_is_str(self):
        assert isinstance(LearningMode.SUPERVISED, str)


# ---------------------------------------------------------------------------
# _PhaseSlots
# ---------------------------------------------------------------------------
class TestPhaseSlots:
    def test_defaults(self):
        ps = _PhaseSlots()
        assert ps.train is None
        assert ps.valid is None
        assert ps.test is None

    def test_iter_active_items_all_none(self):
        ps = _PhaseSlots()
        assert list(ps.iter_active_items()) == []

    def test_iter_active_items_all_set(self):
        ps = _PhaseSlots(train="a", valid="b", test="c")
        items = list(ps.iter_active_items())
        assert items == [("train", "a"), ("valid", "b"), ("test", "c")]

    def test_iter_active_items_partial(self):
        ps = _PhaseSlots(train="a", valid=None, test="c")
        items = list(ps.iter_active_items())
        assert items == [("train", "a"), ("test", "c")]

    def test_iter_active_items_only_valid(self):
        ps = _PhaseSlots(valid="b")
        items = list(ps.iter_active_items())
        assert items == [("valid", "b")]


# ---------------------------------------------------------------------------
# Phase, DataSet, DataLoader (all _PhaseSlots subclasses)
# ---------------------------------------------------------------------------
class TestPhase:
    def test_is_phase_slots(self):
        p = Phase()
        assert isinstance(p, _PhaseSlots)
        assert p.train is None

    def test_set_values(self):
        p = Phase(train="t", valid="v", test="te")
        assert p.train == "t"
        assert p.valid == "v"
        assert p.test == "te"


class TestDataSet:
    def test_is_phase_slots(self):
        ds = DataSet()
        assert isinstance(ds, _PhaseSlots)

    def test_set_values(self):
        ds = DataSet(train="ds_train")
        assert ds.train == "ds_train"


class TestDataLoader:
    def test_is_phase_slots(self):
        dl = DataLoader()
        assert isinstance(dl, _PhaseSlots)


# ---------------------------------------------------------------------------
# _PhaseLists
# ---------------------------------------------------------------------------
class TestPhaseLists:
    def test_default_empty_lists(self):
        pl = _PhaseLists()
        assert pl.train == []
        assert pl.valid == []
        assert pl.test == []

    def test_independent_defaults(self):
        pl1 = _PhaseLists()
        pl2 = _PhaseLists()
        pl1.train.append(1)
        assert pl2.train == []


# ---------------------------------------------------------------------------
# VarName
# ---------------------------------------------------------------------------
class TestVarName:
    def test_defaults(self):
        v = VarName()
        assert v.inputs is None
        assert v.outputs is None
        assert v.true is None

    def test_set_fields(self):
        v = VarName(inputs=["x"], outputs=["y"], true=["t"])
        assert v.inputs == ["x"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_is_phase_lists(self):
        m = Metrics()
        assert isinstance(m, _PhaseLists)
        assert m.train == []
        assert m.valid == []
        assert m.test == []


# ---------------------------------------------------------------------------
# DLEnv
# ---------------------------------------------------------------------------
class TestDLEnv:
    def test_default_values(self):
        env = DLEnv()
        assert isinstance(env.phase, Phase)
        assert isinstance(env.var_name, VarName)
        assert env.learning_mode == LearningMode.SUPERVISED
        assert env.model is None
        assert env.optimizer is None
        assert env.loss is None
        assert env.device is None
        assert isinstance(env.dataset, DataSet)
        assert isinstance(env.dataloader, DataLoader)
        assert isinstance(env.metrics, Metrics)
        assert env.is_compiled is False

    def test_clear(self):
        env = DLEnv()
        env.model = "some_model"
        env.optimizer = "some_opt"
        env.loss = "some_loss"
        env.device = "cpu"
        env.is_compiled = True
        env.learning_mode = LearningMode.UNSUPERVISED
        env.phase.train = "train_phase"

        env.clear()

        assert env.model is None
        assert env.optimizer is None
        assert env.loss is None
        assert env.device is None
        assert env.is_compiled is False
        assert env.learning_mode == LearningMode.SUPERVISED
        assert env.phase.train is None
        assert isinstance(env.dataset, DataSet)
        assert isinstance(env.dataloader, DataLoader)
        assert isinstance(env.metrics, Metrics)
        assert isinstance(env.var_name, VarName)

    # -- require_compiled ---------------------------------------------------
    def test_require_compiled_true(self):
        env = DLEnv()
        env.is_compiled = True
        env.require_compiled("fit")  # should not raise

    def test_require_compiled_false(self):
        env = DLEnv()
        with pytest.raises(RuntimeError, match="Call compile\\(\\) before fit"):
            env.require_compiled("fit")

    # -- require_fit_ready --------------------------------------------------
    def test_require_fit_ready_supervised_all_set(self):
        env = DLEnv()
        env.model = "m"
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.var_name.true = ["y"]
        env.learning_mode = LearningMode.SUPERVISED
        env.require_fit_ready()  # should not raise

    def test_require_fit_ready_supervised_missing_true(self):
        env = DLEnv()
        env.model = "m"
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.learning_mode = LearningMode.SUPERVISED
        # var_name.true is None
        with pytest.raises(ValueError, match="var_name.true"):
            env.require_fit_ready()

    def test_require_fit_ready_self_supervised_requires_true(self):
        env = DLEnv()
        env.model = "m"
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.learning_mode = LearningMode.SELF_SUPERVISED
        # var_name.true is None
        with pytest.raises(ValueError, match="var_name.true"):
            env.require_fit_ready()

    def test_require_fit_ready_self_supervised_ok(self):
        env = DLEnv()
        env.model = "m"
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.var_name.true = ["y"]
        env.learning_mode = LearningMode.SELF_SUPERVISED
        env.require_fit_ready()  # should not raise

    def test_require_fit_ready_unsupervised(self):
        env = DLEnv()
        env.model = "m"
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.learning_mode = LearningMode.UNSUPERVISED
        env.require_fit_ready()  # should not raise (no var_name.true required)

    def test_require_fit_ready_missing_model(self):
        env = DLEnv()
        env.optimizer = "o"
        env.loss = "l"
        env.device = "d"
        env.phase.train = "t"
        env.dataloader.train = "dl"
        env.var_name.true = ["y"]
        with pytest.raises(ValueError, match="model"):
            env.require_fit_ready()

    def test_require_fit_ready_missing_multiple(self):
        env = DLEnv()
        env.learning_mode = LearningMode.SUPERVISED
        with pytest.raises(ValueError, match="model.*optimizer.*loss.*device"):
            env.require_fit_ready()

    # -- require_predict_ready ----------------------------------------------
    def test_require_predict_ready_supervised(self):
        env = DLEnv()
        env.model = "m"
        env.device = "d"
        env.phase.test = "test_phase"
        env.dataloader.test = "dl"
        env.learning_mode = LearningMode.SUPERVISED
        env.require_predict_ready()  # should not raise

    def test_require_predict_ready_self_supervised(self):
        env = DLEnv()
        env.model = "m"
        env.device = "d"
        env.phase.test = "test_phase"
        env.dataloader.test = "dl"
        env.learning_mode = LearningMode.SELF_SUPERVISED
        env.require_predict_ready()

    def test_require_predict_ready_unsupervised(self):
        env = DLEnv()
        env.model = "m"
        env.device = "d"
        env.phase.test = "test_phase"
        env.dataloader.test = "dl"
        env.learning_mode = LearningMode.UNSUPERVISED
        env.require_predict_ready()

    def test_require_predict_ready_missing_model(self):
        env = DLEnv()
        env.device = "d"
        env.phase.test = "test_phase"
        env.dataloader.test = "dl"
        with pytest.raises(ValueError, match="model"):
            env.require_predict_ready()

    def test_require_predict_ready_missing_test_phase(self):
        env = DLEnv()
        env.model = "m"
        env.device = "d"
        env.dataloader.test = "dl"
        with pytest.raises(ValueError, match="phase.test"):
            env.require_predict_ready()

    def test_require_predict_ready_missing_dataloader_test(self):
        env = DLEnv()
        env.model = "m"
        env.device = "d"
        env.phase.test = "test_phase"
        with pytest.raises(ValueError, match="dataloader.test"):
            env.require_predict_ready()

    # -- _require_ready with missing fields ---------------------------------
    def test_require_ready_no_missing(self):
        env = DLEnv()
        env._require_ready("action", (("a", "val"),))

    def test_require_ready_missing(self):
        env = DLEnv()
        with pytest.raises(ValueError, match="action\\(\\) requires: field_a, field_b"):
            env._require_ready("action", (("field_a", None), ("field_b", None), ("field_c", "ok")))
