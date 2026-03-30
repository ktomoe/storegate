from dataclasses import dataclass
import sys
import time
from typing import Any, TextIO, TypeAlias

ShowInfoRow: TypeAlias = tuple[str, str, str, str, str]
_ProgressRow: TypeAlias = tuple[str, str, str, str, str]

_DEFAULT_TABLE_WIDTH = 80
_BATCH_RENDER_INTERVAL = 0.1
_SHOW_INFO_HEADERS = ("Phase", "Variable", "Events", "Dtype", "Shape")
_SHOW_INFO_MIN_WIDTHS = (10, 10, 10, 10, 15)
_PROGRESS_HEADERS = ("Slot", "State", "Job", "Epoch", "Detail")
_PROGRESS_MIN_WIDTHS = (6, 6, 6, 12, 55)


def _build_table(
    title_text: str,
    headers: tuple[str, ...],
    min_widths: tuple[int, ...],
    rows: list[tuple[str, ...]],
    widths: list[int] | None = None,
) -> tuple[str, list[int]]:
    """Render a Unicode box-drawing table and return (table_str, widths)."""
    if widths is None:
        resolved = [
            max(len(h), mw) for h, mw in zip(headers, min_widths)
        ]
    else:
        resolved = list(widths)

    for row in rows:
        for i, cell in enumerate(row):
            resolved[i] = max(resolved[i], len(cell))

    inner_width = sum(w + 3 for w in resolved) - 1
    if len(title_text) > inner_width:
        resolved[-1] += len(title_text) - inner_width

    table_width = sum(w + 3 for w in resolved) + 1
    pad = table_width - 2 - len(title_text)
    left_pad = pad // 2
    right_pad = pad - left_pad

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in resolved) + right

    def data_row(cells: tuple[str, ...]) -> str:
        parts = [f" {c:<{resolved[i]}} " for i, c in enumerate(cells)]
        return "│" + "│".join(parts) + "│"

    lines: list[str] = [
        "┌" + "─" * left_pad + title_text + "─" * right_pad + "┐",
        hline("├", "┬", "┤"),
        data_row(headers),
        hline("├", "┼", "┤"),
    ]
    for row in rows:
        lines.append(data_row(row))
    lines.append(hline("└", "┴", "┘"))

    return "\n".join(lines), resolved


def format_show_info_table(
    *,
    data_id: str,
    backend_label: str | None,
    all_compiled: bool,
    rows: list[ShowInfoRow],
) -> str:
    """Return the show_info() table as a formatted string."""
    status = "Compiled" if all_compiled else "Not Compiled"
    backend_part = f"  backend: {backend_label}" if backend_label else ""
    title = f" data_id: {data_id}{backend_part}  [{status}] "

    display_rows: list[tuple[str, ...]] = []
    prev_phase = ""
    for row in rows:
        display_phase = row[0] if row[0] != prev_phase else ""
        prev_phase = row[0]
        display_rows.append((display_phase,) + row[1:])

    table, _ = _build_table(
        title, _SHOW_INFO_HEADERS, _SHOW_INFO_MIN_WIDTHS, display_rows,
    )
    return table


def format_progress_table(
    *,
    title: str,
    rows: list[_ProgressRow],
    widths: list[int] | None = None,
) -> tuple[str, list[int]]:
    """Return the search progress table and the resolved column widths."""
    return _build_table(
        f" {title} ", _PROGRESS_HEADERS, _PROGRESS_MIN_WIDTHS, rows, widths,
    )


# ---------------------------------------------------------------------------
# Search-agent progress formatter
# ---------------------------------------------------------------------------

@dataclass
class _SlotDisplayState:
    job_id: str | None = None
    job_num: int | None = None
    total: int | None = None
    epoch_text: str = ""
    detail_text: str = ""
    status_text: str = "IDLE"


class ProgressFormatter:
    """Format and print search agent progress."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        interactive: bool | None = None,
    ) -> None:
        self._stream = sys.stdout if stream is None else stream
        if interactive is None:
            self._interactive = bool(
                getattr(self._stream, "isatty", lambda: False)()
            )
        else:
            self._interactive = interactive

        self._slot_order: list[str] = []
        self._slot_states: dict[str, _SlotDisplayState] = {}
        self._render_line_count = 0
        self._table_title = ""
        self._slot_widths = [
            max(len(header), min_width)
            for header, min_width in zip(_PROGRESS_HEADERS, _PROGRESS_MIN_WIDTHS)
        ]
        self._last_batch_render_time = 0.0

    def print_header(
        self,
        total_jobs: int,
        num_combos: int,
        num_trials: int,
        *,
        slot_labels: list[str] | None = None,
    ) -> None:
        self._table_title = (
            f"Search: {total_jobs} jobs "
            f"({num_combos} combos x {num_trials} trials)"
        )

        self._slot_order = list(slot_labels) if slot_labels else ["cpu"]
        self._slot_states = {
            label: _SlotDisplayState()
            for label in self._slot_order
        }
        self._slot_widths = [
            max(len(header), min_width)
            for header, min_width in zip(_PROGRESS_HEADERS, _PROGRESS_MIN_WIDTHS)
        ]

        if self._interactive:
            self._render_slots()
            return

        self._write_plain(
            f"{'=' * _DEFAULT_TABLE_WIDTH}\n"
            f"  {self._table_title}\n"
            f"{'=' * _DEFAULT_TABLE_WIDTH}\n"
        )

    def print_footer(self, n_ok: int, n_err: int) -> None:
        footer = (
            f"{'=' * _DEFAULT_TABLE_WIDTH}\n"
            f"  Done: {n_ok} succeeded, {n_err} failed\n"
            f"{'=' * _DEFAULT_TABLE_WIDTH}\n"
        )
        self._write_plain(footer)

    def print_job_start(
        self,
        slot_label: str,
        job_id: str,
        hps: dict[str, Any],
        job_num: int,
        total: int,
    ) -> None:
        self._slot_states[slot_label] = _SlotDisplayState(
            job_id=job_id,
            job_num=job_num,
            total=total,
            status_text="RUN",
        )

        if self._interactive:
            self._render_slots()
            return

        self._write_plain(
            f"\u25b6 [{slot_label}] [{job_id}] job {job_num}/{total}\n"
        )

    def print_job_end(
        self,
        slot_label: str,
        job_id: str,
        status: str,
        job_num: int,
        total: int,
    ) -> None:
        slot = self._slot_states.setdefault(slot_label, _SlotDisplayState())
        slot.job_id = job_id
        slot.job_num = job_num
        slot.total = total
        slot.status_text = "OK" if status == "success" else "ERR"

        if self._interactive:
            self._render_slots()
            return

        mark = "\u2713" if status == "success" else "\u2717"
        self._write_plain(
            f"  {mark} [{slot_label}] [{job_id}] job {job_num}/{total} {status}\n"
        )

    def print_epoch(
        self,
        slot_label: str,
        job_id: str,
        info: dict[str, Any],
    ) -> None:
        slot = self._slot_states.setdefault(slot_label, _SlotDisplayState())
        slot.job_id = job_id
        slot.epoch_text, slot.detail_text = self._format_epoch(info)
        if slot.status_text == "IDLE":
            slot.status_text = "RUN"

        if self._interactive:
            self._render_slots()
            return

        self._write_plain(
            f"  [{slot_label}] [{job_id}] {slot.epoch_text}\n"
        )

    def print_batch(
        self,
        slot_label: str,
        job_id: str,
        info: dict[str, Any],
    ) -> None:
        slot = self._slot_states.setdefault(slot_label, _SlotDisplayState())
        slot.job_id = job_id

        epoch = info.get("epoch", "?")
        num_epochs = info.get("num_epochs", "?")
        phase = info.get("phase", "")
        batch = info.get("batch", "?")
        total_batches = info.get("total_batches")
        metrics = info.get("metrics", {})

        slot.epoch_text = f"{epoch}/{num_epochs} {phase}".rstrip()

        detail_parts: list[str] = []
        if (
            isinstance(batch, int)
            and isinstance(total_batches, int)
            and total_batches > 0
        ):
            bar_width = 10
            filled = int(bar_width * batch / total_batches)
            bar = "█" * filled + "░" * (bar_width - filled)
            detail_parts.append(f"{bar} {batch}/{total_batches}")
        else:
            detail_parts.append(f"batch {batch}")

        metric_strs = [
            f"{k}={v:.4f}"
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ]
        if metric_strs:
            detail_parts.append(" ".join(metric_strs))

        batch_rate = info.get("batch_rate")
        if isinstance(batch_rate, (int, float)) and batch_rate > 0:
            detail_parts.append(f"{batch_rate:.1f} batch/s")

        slot.detail_text = "  ".join(detail_parts)

        if slot.status_text == "IDLE":
            slot.status_text = "RUN"

        if not self._interactive:
            if (
                isinstance(batch, int)
                and isinstance(total_batches, int)
                and batch >= total_batches
            ):
                self._write_plain(
                    f"  [{slot_label}] [{job_id}] {slot.epoch_text}\n"
                )
            return

        now = time.monotonic()
        if now - self._last_batch_render_time < _BATCH_RENDER_INTERVAL:
            return
        self._last_batch_render_time = now
        self._render_slots()

    # -- internal -------------------------------------------------------------

    def _format_epoch(self, info: dict[str, Any]) -> tuple[str, str]:
        epoch = info.get("epoch", "?")
        num_epochs = info.get("num_epochs", "?")
        phase = info.get("phase", "")
        metrics = info.get("metrics", {})

        epoch_text = f"{epoch}/{num_epochs} {phase}".rstrip()
        detail_parts: list[str] = []

        metric_strs = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_strs.append(f"{k}={v:.4f}")
        if metric_strs:
            detail_parts.append(" | ".join(metric_strs))

        rate = info.get("batch_rate")
        if rate and rate > 0:
            detail_parts.append(f"{rate:.1f} batch/s")

        elapsed = info.get("elapsed")
        if elapsed is not None:
            detail_parts.append(f"{elapsed:.1f}s")

        return epoch_text, "  ".join(detail_parts)

    def _format_slot_row(
        self,
        slot_label: str,
        slot: _SlotDisplayState,
    ) -> _ProgressRow:
        job_text = ""
        if slot.job_num is not None and slot.total is not None:
            job_text = f"{slot.job_num}/{slot.total}"

        return (
            slot_label,
            slot.status_text,
            job_text,
            slot.epoch_text,
            slot.detail_text,
        )

    def _render_slots(self) -> None:
        rows = [
            self._format_slot_row(label, self._slot_states[label])
            for label in self._slot_order
        ]
        block, self._slot_widths = format_progress_table(
            title=self._table_title,
            rows=rows,
            widths=self._slot_widths,
        )

        if self._render_line_count > 0:
            self._stream.write(f"\033[{self._render_line_count}F")
            self._stream.write("\r\033[J")

        self._stream.write(block + "\n")
        self._stream.flush()
        self._render_line_count = len(block.splitlines())

    def _write_plain(self, msg: str) -> None:
        self._stream.write(msg)
        self._stream.flush()
