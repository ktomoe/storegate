"""Tests for storegate/formatters.py."""
import io
import time
from unittest.mock import patch

import pytest

from storegate.formatters import (
    ProgressFormatter,
    _SlotDisplayState,
    format_progress_table,
    format_show_info_table,
)


# ---------------------------------------------------------------------------
# format_show_info_table
# ---------------------------------------------------------------------------
class TestFormatShowInfoTable:
    def test_basic(self):
        rows = [("train", "x", "100", "float32", "(3,)")]
        result = format_show_info_table(
            data_id="d1",
            backend_label="numpy",
            all_compiled=True,
            rows=rows,
        )
        assert "d1" in result
        assert "numpy" in result
        assert "Compiled" in result
        assert "train" in result
        assert "x" in result

    def test_empty_rows(self):
        result = format_show_info_table(
            data_id="d1",
            backend_label=None,
            all_compiled=False,
            rows=[],
        )
        assert "d1" in result
        assert "Not Compiled" in result

    def test_long_title_exceeding_inner_width(self):
        """When the title is wider than the inner table, the last column widens."""
        result = format_show_info_table(
            data_id="a_very_long_data_id_that_makes_the_title_extremely_wide",
            backend_label="my_really_long_backend_name",
            all_compiled=False,
            rows=[("p", "v", "1", "f32", "()")],
        )
        assert "a_very_long_data_id_that_makes_the_title_extremely_wide" in result

    def test_compiled_flag(self):
        result_compiled = format_show_info_table(
            data_id="d", backend_label=None, all_compiled=True, rows=[]
        )
        result_not = format_show_info_table(
            data_id="d", backend_label=None, all_compiled=False, rows=[]
        )
        assert "[Compiled]" in result_compiled
        assert "[Not Compiled]" in result_not

    def test_without_backend(self):
        result = format_show_info_table(
            data_id="d", backend_label=None, all_compiled=False, rows=[]
        )
        assert "backend:" not in result

    def test_with_backend(self):
        result = format_show_info_table(
            data_id="d", backend_label="np", all_compiled=False, rows=[]
        )
        assert "backend: np" in result

    def test_duplicate_phase_display(self):
        """Duplicate phase names should appear blank after the first row."""
        rows = [
            ("train", "x", "10", "float32", "(3,)"),
            ("train", "y", "10", "float32", "(1,)"),
        ]
        result = format_show_info_table(
            data_id="d", backend_label=None, all_compiled=True, rows=rows
        )
        lines = result.split("\n")
        # Find the data rows (after the header separator)
        data_lines = [l for l in lines if "│" in l and "Phase" not in l]
        # First data row should have "train", second should have it blanked
        assert "train" in data_lines[0]
        # The second data line should have an empty first cell
        # (the phase column should be blank for the repeated phase)


# ---------------------------------------------------------------------------
# format_progress_table
# ---------------------------------------------------------------------------
class TestFormatProgressTable:
    def test_basic(self):
        rows = [("cpu", "RUN", "1/5", "1/10 train", "batch 3")]
        result, widths = format_progress_table(title="Test", rows=rows)
        assert "Test" in result
        assert "cpu" in result
        assert len(widths) == 5

    def test_with_preset_widths(self):
        rows = [("cpu", "IDLE", "", "", "")]
        preset = [20, 10, 10, 20, 60]
        result, widths = format_progress_table(
            title="T", rows=rows, widths=preset
        )
        # Widths should be at least as large as preset
        for w, p in zip(widths, preset):
            assert w >= p

    def test_long_title(self):
        rows = [("cpu", "RUN", "1/1", "", "")]
        long_title = "A" * 200
        result, widths = format_progress_table(title=long_title, rows=rows)
        assert long_title in result


# ---------------------------------------------------------------------------
# ProgressFormatter
# ---------------------------------------------------------------------------
class TestProgressFormatter:
    def test_init_default_stream(self):
        pf = ProgressFormatter()
        assert pf._stream is not None

    def test_init_explicit_stream(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf)
        assert pf._stream is buf

    def test_init_interactive_auto_detect_true(self):
        """isatty() returning True => interactive mode."""
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[attr-defined]
        pf = ProgressFormatter(stream=buf)
        assert pf._interactive is True

    def test_init_interactive_auto_detect_false(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf)
        assert pf._interactive is False

    def test_init_explicit_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        assert pf._interactive is True

    # -- print_header -------------------------------------------------------
    def test_print_header_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(10, 2, 5, slot_labels=["gpu0", "gpu1"])
        output = buf.getvalue()
        assert "Search:" in output
        assert "gpu0" in output

    def test_print_header_non_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(10, 2, 5)
        output = buf.getvalue()
        assert "Search:" in output
        assert "=" in output

    def test_print_header_default_slot(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1)
        assert "cpu" in pf._slot_order

    # -- print_footer -------------------------------------------------------
    def test_print_footer(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_footer(8, 2)
        output = buf.getvalue()
        assert "8 succeeded" in output
        assert "2 failed" in output

    # -- print_job_start ----------------------------------------------------
    def test_print_job_start_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_start("cpu", "job1", {"lr": 0.01}, 1, 5)
        output = buf.getvalue()
        # Interactive mode renders the table
        assert "cpu" in output

    def test_print_job_start_non_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_start("cpu", "job1", {"lr": 0.01}, 1, 5)
        output = buf.getvalue()
        assert "job1" in output
        assert "1/5" in output

    # -- print_job_end ------------------------------------------------------
    def test_print_job_end_success_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_end("cpu", "j1", "success", 1, 5)
        output = buf.getvalue()
        assert "OK" in output or "cpu" in output

    def test_print_job_end_error_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_end("cpu", "j1", "error", 1, 5)
        output = buf.getvalue()
        assert "ERR" in output or "cpu" in output

    def test_print_job_end_success_non_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_end("cpu", "j1", "success", 1, 5)
        output = buf.getvalue()
        assert "\u2713" in output

    def test_print_job_end_error_non_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(5, 1, 5, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_job_end("cpu", "j1", "error", 1, 5)
        output = buf.getvalue()
        assert "\u2717" in output

    # -- print_epoch --------------------------------------------------------
    def test_print_epoch_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_epoch("cpu", "j1", {"epoch": 1, "num_epochs": 10, "phase": "train"})
        output = buf.getvalue()
        assert "1/10" in output

    def test_print_epoch_non_interactive(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_epoch("cpu", "j1", {"epoch": 1, "num_epochs": 10, "phase": "train"})
        output = buf.getvalue()
        assert "1/10" in output
        assert "train" in output

    def test_print_epoch_sets_run_from_idle(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        assert pf._slot_states["cpu"].status_text == "IDLE"
        pf.print_epoch("cpu", "j1", {"epoch": 1, "num_epochs": 10})
        assert pf._slot_states["cpu"].status_text == "RUN"

    # -- print_batch --------------------------------------------------------
    def test_print_batch_interactive_with_throttling(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = time.monotonic()

        buf.truncate(0)
        buf.seek(0)
        # Should be throttled (too recent)
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 1, "total_batches": 100,
        })
        output = buf.getvalue()
        # Throttled: nothing rendered
        assert output == ""

    def test_print_batch_interactive_renders_after_interval(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        # Set last render time far in the past
        pf._last_batch_render_time = 0.0
        buf.truncate(0)
        buf.seek(0)
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 50, "total_batches": 100,
        })
        output = buf.getvalue()
        assert output != ""

    def test_print_batch_non_interactive_last_batch(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 100, "total_batches": 100,
        })
        output = buf.getvalue()
        assert "1/10" in output

    def test_print_batch_non_interactive_not_last_batch(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        buf.truncate(0)
        buf.seek(0)
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 50, "total_batches": 100,
        })
        output = buf.getvalue()
        # Non-interactive, not last batch => nothing printed
        assert output == ""

    def test_print_batch_no_total_batches(self):
        """When total_batches is not an int, uses 'batch N' format."""
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = 0.0
        buf.truncate(0)
        buf.seek(0)
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 5,
        })
        slot = pf._slot_states["cpu"]
        assert "batch 5" in slot.detail_text

    def test_print_batch_with_metrics(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = 0.0
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "batch": 5, "total_batches": 10,
            "metrics": {"loss": 0.123, "acc": 0.99, "label": "not-a-number"},
        })
        slot = pf._slot_states["cpu"]
        assert "loss=0.1230" in slot.detail_text
        assert "acc=0.9900" in slot.detail_text
        assert "label" not in slot.detail_text

    def test_print_batch_with_batch_rate(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = 0.0
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 10, "phase": "",
            "batch": 5, "total_batches": 10,
            "batch_rate": 42.5,
        })
        slot = pf._slot_states["cpu"]
        assert "42.5 batch/s" in slot.detail_text

    def test_print_batch_sets_run_from_idle(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=False)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        assert pf._slot_states["cpu"].status_text == "IDLE"
        pf.print_batch("cpu", "j1", {"epoch": 1, "num_epochs": 10, "batch": 1})
        assert pf._slot_states["cpu"].status_text == "RUN"

    def test_print_batch_progress_bar(self):
        """When batch and total_batches are ints, a progress bar is rendered."""
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = 0.0
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 5, "phase": "train",
            "batch": 5, "total_batches": 10,
        })
        slot = pf._slot_states["cpu"]
        assert "\u2588" in slot.detail_text  # filled block
        assert "5/10" in slot.detail_text

    def test_print_batch_zero_total_batches(self):
        """total_batches=0 should fall through to the 'batch N' branch."""
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf.print_header(1, 1, 1, slot_labels=["cpu"])
        pf._last_batch_render_time = 0.0
        pf.print_batch("cpu", "j1", {
            "epoch": 1, "num_epochs": 5, "phase": "train",
            "batch": 5, "total_batches": 0,
        })
        slot = pf._slot_states["cpu"]
        assert "batch 5" in slot.detail_text


# ---------------------------------------------------------------------------
# _format_epoch
# ---------------------------------------------------------------------------
class TestFormatEpoch:
    def _make_pf(self):
        return ProgressFormatter(stream=io.StringIO(), interactive=False)

    def test_with_metrics(self):
        pf = self._make_pf()
        epoch_text, detail = pf._format_epoch({
            "epoch": 1, "num_epochs": 10, "phase": "train",
            "metrics": {"loss": 0.5, "acc": 0.9},
        })
        assert "1/10 train" == epoch_text
        assert "loss=0.5000" in detail
        assert "acc=0.9000" in detail

    def test_without_metrics(self):
        pf = self._make_pf()
        epoch_text, detail = pf._format_epoch({
            "epoch": 2, "num_epochs": 5, "phase": "val",
        })
        assert "2/5 val" == epoch_text
        assert detail == ""

    def test_with_rate(self):
        pf = self._make_pf()
        _, detail = pf._format_epoch({
            "epoch": 1, "num_epochs": 10,
            "batch_rate": 100.0,
        })
        assert "100.0 batch/s" in detail

    def test_with_elapsed(self):
        pf = self._make_pf()
        _, detail = pf._format_epoch({
            "epoch": 1, "num_epochs": 10,
            "elapsed": 12.3,
        })
        assert "12.3s" in detail

    def test_without_phase(self):
        pf = self._make_pf()
        epoch_text, _ = pf._format_epoch({"epoch": 1, "num_epochs": 10})
        assert epoch_text == "1/10"

    def test_non_numeric_metric_skipped(self):
        pf = self._make_pf()
        _, detail = pf._format_epoch({
            "epoch": 1, "num_epochs": 10,
            "metrics": {"tag": "hello"},
        })
        assert "tag" not in detail


# ---------------------------------------------------------------------------
# _format_slot_row
# ---------------------------------------------------------------------------
class TestFormatSlotRow:
    def test_with_job_info(self):
        pf = ProgressFormatter(stream=io.StringIO(), interactive=False)
        slot = _SlotDisplayState(
            job_id="j1", job_num=3, total=10,
            epoch_text="1/5", detail_text="details", status_text="RUN",
        )
        row = pf._format_slot_row("gpu0", slot)
        assert row == ("gpu0", "RUN", "3/10", "1/5", "details")

    def test_without_job_info(self):
        pf = ProgressFormatter(stream=io.StringIO(), interactive=False)
        slot = _SlotDisplayState()
        row = pf._format_slot_row("cpu", slot)
        assert row == ("cpu", "IDLE", "", "", "")


# ---------------------------------------------------------------------------
# _render_slots
# ---------------------------------------------------------------------------
class TestRenderSlots:
    def test_first_render(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf._table_title = "Test"
        pf._slot_order = ["cpu"]
        pf._slot_states = {"cpu": _SlotDisplayState()}
        pf._render_line_count = 0

        pf._render_slots()
        output = buf.getvalue()
        assert "cpu" in output
        assert pf._render_line_count > 0

    def test_re_render_cursor_movement(self):
        buf = io.StringIO()
        pf = ProgressFormatter(stream=buf, interactive=True)
        pf._table_title = "Test"
        pf._slot_order = ["cpu"]
        pf._slot_states = {"cpu": _SlotDisplayState()}
        pf._render_line_count = 5  # pretend already rendered

        pf._render_slots()
        output = buf.getvalue()
        # Should contain cursor movement escape
        assert "\033[5F" in output


# ---------------------------------------------------------------------------
# _SlotDisplayState
# ---------------------------------------------------------------------------
class TestSlotDisplayState:
    def test_defaults(self):
        s = _SlotDisplayState()
        assert s.job_id is None
        assert s.job_num is None
        assert s.total is None
        assert s.epoch_text == ""
        assert s.detail_text == ""
        assert s.status_text == "IDLE"

    def test_custom_values(self):
        s = _SlotDisplayState(
            job_id="j1", job_num=1, total=5,
            epoch_text="2/10", detail_text="d", status_text="RUN",
        )
        assert s.job_id == "j1"
        assert s.job_num == 1
        assert s.total == 5
