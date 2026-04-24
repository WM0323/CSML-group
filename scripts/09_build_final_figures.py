#!/usr/bin/env python3
"""Build final report/presentation figures without adding plotting dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

WALKFORWARD_LABELS_PATH = (
    PROJECT_ROOT / "results" / "regimes" / "hmm_walkforward_state_labels.parquet"
)
WALKFORWARD_PROBABILITIES_PATH = (
    PROJECT_ROOT / "results" / "regimes" / "hmm_walkforward_state_probabilities.parquet"
)
KMEANS_LABELS_PATH = PROJECT_ROOT / "results" / "regimes" / "cluster_labels.parquet"
WALKFORWARD_STRATEGY_PATH = (
    PROJECT_ROOT / "results" / "backtests" / "walkforward_strategy_returns.parquet"
)
WALKFORWARD_BENCHMARKS_PATH = (
    PROJECT_ROOT / "results" / "backtests" / "walkforward_benchmark_returns.parquet"
)
KMEANS_STRATEGY_PATH = PROJECT_ROOT / "results" / "backtests" / "kmeans_strategy_returns.parquet"
COMPARISON_METRICS_PATH = (
    PROJECT_ROOT / "results" / "backtests" / "kmeans_vs_walkforward_strategy_metrics.csv"
)

PALETTE = {
    "bg": "#F7F2EA",
    "panel": "#FFFDF9",
    "ink": "#182126",
    "muted": "#6F7D85",
    "grid": "#DCCFBE",
    "panel_line": "#CDBFAF",
    "hmm": "#176B59",
    "kmeans": "#C56C28",
    "equal": "#4D6FD7",
    "sixty_forty": "#8E5BB8",
    "state_0": "#D95F02",
    "state_1": "#F1A340",
    "state_2": "#2B8CBE",
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    preferred_paths = []
    if bold:
        preferred_paths.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    preferred_paths.extend(FONT_CANDIDATES)
    for path in preferred_paths:
        candidate = Path(path)
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


TITLE_FONT = load_font(42, bold=True)
SUBTITLE_FONT = load_font(20)
SECTION_FONT = load_font(24, bold=True)
LABEL_FONT = load_font(20)
SMALL_FONT = load_font(16)
VALUE_FONT = load_font(22, bold=True)


def ensure_figure_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def infer_walkforward_comparison_row(metrics: pd.DataFrame) -> str:
    candidates = [
        portfolio_name
        for portfolio_name in metrics["portfolio_name"].astype(str).tolist()
        if portfolio_name.startswith("walkforward_hmm")
    ]
    if not candidates:
        raise ValueError("Comparison metrics are missing a walk-forward HMM row.")
    return candidates[0]


def compute_run_statistics(labels: pd.Series) -> tuple[float, float]:
    values = labels.astype(int).to_numpy()
    same_as_previous = float((values[1:] == values[:-1]).mean()) if len(values) > 1 else 1.0

    run_lengths: list[int] = []
    current = int(values[0])
    length = 1
    for value in values[1:]:
        value_int = int(value)
        if value_int == current:
            length += 1
        else:
            run_lengths.append(length)
            current = value_int
            length = 1
    run_lengths.append(length)
    return same_as_previous, float(np.mean(run_lengths))


def new_canvas(width: int = 1600, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), PALETTE["bg"])
    return image, ImageDraw.Draw(image)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((72, 46), title, font=TITLE_FONT, fill=PALETTE["ink"])
    draw.text((72, 104), subtitle, font=SUBTITLE_FONT, fill=PALETTE["muted"])


def draw_footer(draw: ImageDraw.ImageDraw, text: str, canvas_height: int) -> None:
    draw.text((72, canvas_height - 46), text, font=SMALL_FONT, fill=PALETTE["muted"])


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(
        box,
        radius=22,
        fill=PALETTE["panel"],
        outline=PALETTE["panel_line"],
        width=2,
    )
    draw.text((left + 24, top + 18), title, font=SECTION_FONT, fill=PALETTE["ink"])
    draw.line((left + 24, top + 58, right - 24, top + 58), fill=PALETTE["grid"], width=2)


def draw_wrapped_legend(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    items: list[tuple[str, str]],
    *,
    start_y: int,
    start_x_offset: int = 70,
    item_gap: int = 28,
    row_gap: int = 14,
) -> int:
    left, _, right, _ = box
    x = left + start_x_offset
    y = start_y
    swatch_width = 24
    swatch_height = 12
    text_y_offset = -6
    row_height = 24
    max_x = right - 34

    for label, color in items:
        text_bbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
        text_width = text_bbox[2] - text_bbox[0]
        item_width = swatch_width + 10 + text_width
        if x != left + start_x_offset and x + item_width > max_x:
            x = left + start_x_offset
            y += row_height + row_gap

        draw.rounded_rectangle(
            (x, y + 6, x + swatch_width, y + 6 + swatch_height),
            radius=4,
            fill=color,
        )
        draw.text((x + swatch_width + 10, y + text_y_offset), label, font=LABEL_FONT, fill=PALETTE["ink"])
        x += item_width + item_gap

    return y + row_height


def draw_vertical_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    values: list[float],
    colors: list[str],
    ymax: float,
    formatter,
) -> None:
    left, top, right, bottom = box
    chart_left = left + 60
    chart_right = right - 30
    chart_top = top + 84
    chart_bottom = bottom - 70

    for tick in range(6):
        y = chart_bottom - (chart_bottom - chart_top) * tick / 5
        draw.line((chart_left, y, chart_right, y), fill=PALETTE["grid"], width=1)
        tick_value = ymax * tick / 5
        text = formatter(tick_value)
        draw.text((left + 12, y - 10), text, font=SMALL_FONT, fill=PALETTE["muted"])

    bar_width = int((chart_right - chart_left) / (len(values) * 2.2))
    gap = int(bar_width * 1.1)
    start_x = chart_left + gap
    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        bar_left = start_x + idx * (bar_width + gap)
        bar_right = bar_left + bar_width
        bar_height = 0 if ymax <= 0 else (chart_bottom - chart_top) * value / ymax
        bar_top = chart_bottom - bar_height
        draw.rounded_rectangle(
            (bar_left, bar_top, bar_right, chart_bottom),
            radius=12,
            fill=color,
        )
        value_text = formatter(value)
        value_width = draw.textbbox((0, 0), value_text, font=VALUE_FONT)[2]
        draw.text(
            (bar_left + (bar_width - value_width) / 2, bar_top - 32),
            value_text,
            font=VALUE_FONT,
            fill=PALETTE["ink"],
        )
        label_width = draw.textbbox((0, 0), label, font=LABEL_FONT)[2]
        draw.text(
            (bar_left + (bar_width - label_width) / 2, chart_bottom + 18),
            label,
            font=LABEL_FONT,
            fill=PALETTE["ink"],
        )


def draw_line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    frame: pd.DataFrame,
    date_column: str,
    series_specs: list[tuple[str, str, str]],
    y_label: str,
) -> None:
    left, top, right, bottom = box
    chart_left = left + 70
    chart_right = right - 34
    legend_bottom = draw_wrapped_legend(
        draw,
        box,
        [(label, color) for label, _, color in series_specs],
        start_y=top + 74,
    )
    chart_top = legend_bottom + 28
    chart_bottom = bottom - 70

    y_values = []
    for _, column, _ in series_specs:
        y_values.extend(frame[column].tolist())
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = (y_max - y_min) * 0.08
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    def map_y(value: float) -> int:
        if np.isclose(y_max, y_min):
            return int((chart_top + chart_bottom) / 2)
        return int(chart_bottom - (value - y_min) * (chart_bottom - chart_top) / (y_max - y_min))

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = map_y(value)
        draw.line((chart_left, y, chart_right, y), fill=PALETTE["grid"], width=1)
        draw.text((left + 12, y - 10), f"{value:.2f}", font=SMALL_FONT, fill=PALETTE["muted"])

    min_date = frame[date_column].min()
    max_date = frame[date_column].max()
    total_days = max((max_date - min_date).days, 1)

    def map_x(timestamp: pd.Timestamp) -> int:
        delta = (timestamp - min_date).days
        return int(chart_left + delta * (chart_right - chart_left) / total_days)

    year_ticks = pd.date_range(min_date.normalize(), max_date.normalize(), freq="2YS")
    if len(year_ticks) == 0 or year_ticks[-1].year != max_date.year:
        year_ticks = year_ticks.append(pd.DatetimeIndex([max_date.normalize()]))
    for timestamp in year_ticks:
        x = map_x(timestamp)
        draw.line((x, chart_top, x, chart_bottom), fill=PALETTE["grid"], width=1)
        label = timestamp.strftime("%Y")
        label_width = draw.textbbox((0, 0), label, font=SMALL_FONT)[2]
        draw.text((x - label_width / 2, chart_bottom + 18), label, font=SMALL_FONT, fill=PALETTE["muted"])

    for _, column, color in series_specs:
        points = [(map_x(ts), map_y(value)) for ts, value in zip(frame[date_column], frame[column], strict=True)]
        draw.line(points, fill=color, width=4)

    draw.text((chart_left, chart_top - 36), y_label, font=SMALL_FONT, fill=PALETTE["muted"])


def draw_probability_ribbon(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    probabilities: pd.DataFrame,
) -> None:
    left, top, right, bottom = box
    chart_left = left + 60
    chart_right = right - 30
    legend_bottom = draw_wrapped_legend(
        draw,
        box,
        [
            ("State 0: risk-off", PALETTE["state_0"]),
            ("State 1: transition", PALETTE["state_1"]),
            ("State 2: risk-on", PALETTE["state_2"]),
        ],
        start_y=top + 74,
    )
    chart_top = legend_bottom + 24
    chart_bottom = bottom - 60

    reduced = probabilities.copy()
    target_points = max(300, chart_right - chart_left)
    if len(reduced) > target_points:
        bins = np.linspace(0, len(reduced), target_points + 1, dtype=int)
        rows = []
        for start, end in zip(bins[:-1], bins[1:], strict=True):
            block = reduced.iloc[start:end]
            if block.empty:
                continue
            rows.append(
                {
                    "date": block["date"].iloc[-1],
                    "state_0_probability": float(block["state_0_probability"].mean()),
                    "state_1_probability": float(block["state_1_probability"].mean()),
                    "state_2_probability": float(block["state_2_probability"].mean()),
                }
            )
        reduced = pd.DataFrame(rows)

    width = max(chart_right - chart_left, 1)
    for idx, row in reduced.reset_index(drop=True).iterrows():
        x = int(chart_left + idx * width / max(len(reduced) - 1, 1))
        y0 = chart_bottom
        h0 = int((chart_bottom - chart_top) * float(row["state_0_probability"]))
        h1 = int((chart_bottom - chart_top) * float(row["state_1_probability"]))
        h2 = int((chart_bottom - chart_top) * float(row["state_2_probability"]))
        draw.line((x, y0, x, y0 - h0), fill=PALETTE["state_0"], width=2)
        draw.line((x, y0 - h0, x, y0 - h0 - h1), fill=PALETTE["state_1"], width=2)
        draw.line((x, y0 - h0 - h1, x, y0 - h0 - h1 - h2), fill=PALETTE["state_2"], width=2)

    for tick in range(6):
        y = chart_bottom - (chart_bottom - chart_top) * tick / 5
        draw.line((chart_left, y, chart_right, y), fill=PALETTE["grid"], width=1)
        draw.text((left + 10, y - 10), f"{tick * 20}%", font=SMALL_FONT, fill=PALETTE["muted"])

    min_date = probabilities["date"].min()
    max_date = probabilities["date"].max()
    year_ticks = pd.date_range(min_date.normalize(), max_date.normalize(), freq="YS")
    if len(year_ticks) == 0 or year_ticks[-1].year != max_date.year:
        year_ticks = year_ticks.append(pd.DatetimeIndex([max_date.normalize()]))
    total_days = max((max_date - min_date).days, 1)
    for timestamp in year_ticks:
        delta = (timestamp - min_date).days
        x = int(chart_left + delta * (chart_right - chart_left) / total_days)
        label = timestamp.strftime("%Y")
        label_width = draw.textbbox((0, 0), label, font=SMALL_FONT)[2]
        draw.text((x - label_width / 2, chart_bottom + 14), label, font=SMALL_FONT, fill=PALETTE["muted"])


def draw_horizontal_metric_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    formatter,
) -> None:
    draw_panel(draw, box, title)
    left, top, right, bottom = box
    chart_top = top + 90
    row_gap = 64
    scale_max = max(values)
    label_font = LABEL_FONT
    label_width = max(draw.textbbox((0, 0), label, font=label_font)[2] for label in labels)
    value_width = max(draw.textbbox((0, 0), formatter(value), font=VALUE_FONT)[2] for value in values)

    panel_width = right - left
    label_gap = 18
    value_gutter = max(88, value_width + 20)
    min_track_width = 92
    available_width = panel_width - 48
    max_label_area = available_width - value_gutter - min_track_width - label_gap
    if label_width > max_label_area:
        label_font = SMALL_FONT
        label_width = max(draw.textbbox((0, 0), label, font=label_font)[2] for label in labels)
    label_area = min(max(150, label_width + 18), max_label_area)

    chart_left = left + 24 + label_area + label_gap
    track_right = right - 24 - value_gutter
    chart_right = right - 24

    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = chart_top + idx * row_gap
        draw.text((left + 24, y + 8), label, font=label_font, fill=PALETTE["ink"])
        x0 = chart_left
        x1 = int(chart_left + value * (track_right - chart_left) / max(scale_max, 1e-9))
        draw.rounded_rectangle((chart_left, y, track_right, y + 34), radius=12, fill="#F2EADF")
        draw.rounded_rectangle((x0, y, x1, y + 34), radius=12, fill=color)
        value_text = formatter(value)
        text_box = draw.textbbox((0, 0), value_text, font=VALUE_FONT)
        text_width = text_box[2] - text_box[0]
        if x1 - x0 >= text_width + 28:
            text_x = x1 - text_width - 14
            text_fill = PALETTE["panel"]
        else:
            text_x = min(x1 + 12, chart_right - text_width)
            text_fill = PALETTE["ink"]
        draw.text((text_x, y + 8), value_text, font=VALUE_FONT, fill=text_fill)


def build_persistence_figure() -> Path:
    hmm_labels = normalize_dates(pd.read_parquet(WALKFORWARD_LABELS_PATH))
    kmeans_labels = normalize_dates(pd.read_parquet(KMEANS_LABELS_PATH))
    overlap = hmm_labels.merge(kmeans_labels, on="date", how="inner", validate="one_to_one")
    hmm_same, hmm_run = compute_run_statistics(overlap["hmm_state"])
    kmeans_same, kmeans_run = compute_run_statistics(overlap["cluster_label"])
    window_text = (
        f"Window: {overlap['date'].min().date().isoformat()} to "
        f"{overlap['date'].max().date().isoformat()}"
    )

    image, draw = new_canvas()
    draw_header(
        draw,
        "Shared-Window Regime Stability",
        "The walk-forward HMM remains materially more persistent than the walk-forward K-means baseline.",
    )
    left_panel = (72, 170, 776, 760)
    right_panel = (824, 170, 1528, 760)
    draw_panel(draw, left_panel, "Same-As-Previous-Day Rate")
    draw_panel(draw, right_panel, "Average Run Length")

    draw_vertical_bars(
        draw,
        left_panel,
        ["Walk-forward\nHMM", "K-means"],
        [hmm_same * 100.0, kmeans_same * 100.0],
        [PALETTE["hmm"], PALETTE["kmeans"]],
        ymax=100.0,
        formatter=lambda value: f"{value:.1f}%",
    )
    draw_vertical_bars(
        draw,
        right_panel,
        ["Walk-forward\nHMM", "K-means"],
        [hmm_run, kmeans_run],
        [PALETTE["hmm"], PALETTE["kmeans"]],
        ymax=max(hmm_run, kmeans_run) * 1.25,
        formatter=lambda value: f"{value:.2f}",
    )
    draw_footer(draw, window_text, image.height)

    output_path = FIGURES_DIR / "figure_01_regime_persistence_comparison.png"
    image.save(output_path)
    return output_path


def build_nav_figure() -> Path:
    walkforward_strategy = normalize_dates(pd.read_parquet(WALKFORWARD_STRATEGY_PATH))
    walkforward_benchmarks = normalize_dates(pd.read_parquet(WALKFORWARD_BENCHMARKS_PATH))
    kmeans_strategy = normalize_dates(pd.read_parquet(KMEANS_STRATEGY_PATH))

    merged = (
        walkforward_strategy[["date", "strategy_nav"]]
        .rename(columns={"strategy_nav": "walkforward_hmm_nav"})
        .merge(
            kmeans_strategy[["date", "strategy_nav"]].rename(columns={"strategy_nav": "kmeans_nav"}),
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .merge(walkforward_benchmarks, on="date", how="inner", validate="one_to_one")
    )
    window_text = (
        f"Window: {merged['date'].min().date().isoformat()} to "
        f"{merged['date'].max().date().isoformat()}"
    )

    image, draw = new_canvas()
    draw_header(
        draw,
        "Shared-Window NAV Comparison",
        "The refreshed HMM portfolio layer materially beats K-means and equal-weight, while fixed 60/40 still finishes with the highest shared-window NAV.",
    )
    panel = (72, 170, 1528, 782)
    draw_panel(draw, panel, "Growth of $1")
    draw_line_chart(
        draw,
        panel,
        merged,
        "date",
        [
            ("Walk-forward HMM", "walkforward_hmm_nav", PALETTE["hmm"]),
            ("K-means", "kmeans_nav", PALETTE["kmeans"]),
            ("Equal-weight 4-asset", "equal_weight_4_asset_nav", PALETTE["equal"]),
            ("Fixed 60/40", "fixed_60_40_stock_bond_nav", PALETTE["sixty_forty"]),
        ],
        y_label="Net asset value",
    )
    draw_footer(draw, window_text, image.height)

    output_path = FIGURES_DIR / "figure_02_shared_window_nav_comparison.png"
    image.save(output_path)
    return output_path


def build_probability_figure() -> Path:
    probabilities = normalize_dates(pd.read_parquet(WALKFORWARD_PROBABILITIES_PATH))
    recent = probabilities.loc[probabilities["date"] >= pd.Timestamp("2019-01-01")].copy()

    image, draw = new_canvas()
    draw_header(
        draw,
        "Walk-Forward HMM Predictive State Probabilities",
        "The tradable signal uses the predictive distribution P(z_t | x_1:t-1), shown here from 2019 onward for readability.",
    )
    panel = (72, 170, 1528, 782)
    draw_panel(draw, panel, "Predictive Probability Ribbon")
    draw_probability_ribbon(draw, panel, recent)
    draw_footer(draw, "Orange = risk-off, amber = transition, blue = risk-on", image.height)

    output_path = FIGURES_DIR / "figure_03_walkforward_state_probabilities.png"
    image.save(output_path)
    return output_path


def build_metric_figure() -> Path:
    metrics = pd.read_csv(COMPARISON_METRICS_PATH)
    walkforward_row_name = infer_walkforward_comparison_row(metrics)
    metrics = metrics.set_index("portfolio_name").loc[
        [
            walkforward_row_name,
            "kmeans_regime_strategy",
            "equal_weight_4_asset",
            "fixed_60_40_stock_bond",
        ]
    ]
    labels = ["Walk-forward HMM", "K-means", "Equal-weight", "Fixed 60/40"]
    colors = [PALETTE["hmm"], PALETTE["kmeans"], PALETTE["equal"], PALETTE["sixty_forty"]]
    walkforward_strategy = normalize_dates(pd.read_parquet(WALKFORWARD_STRATEGY_PATH))
    kmeans_strategy = normalize_dates(pd.read_parquet(KMEANS_STRATEGY_PATH))
    shared_window = walkforward_strategy[["date"]].merge(
        kmeans_strategy[["date"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    window_text = (
        "All values computed on the shared "
        f"{shared_window['date'].min().date().isoformat()} to "
        f"{shared_window['date'].max().date().isoformat()} window."
    )

    image, draw = new_canvas(height=980)
    draw_header(
        draw,
        "Shared-Window Strategy Metrics",
        "The primary HMM strategy now beats K-means and equal-weight by a wide margin, and it edges fixed 60/40 on Sharpe while trailing it slightly on cumulative return.",
    )

    panel_1 = (72, 170, 500, 860)
    panel_2 = (536, 170, 964, 860)
    panel_3 = (1000, 170, 1528, 860)

    draw_horizontal_metric_bars(
        draw,
        panel_1,
        "Cumulative Return",
        labels,
        (metrics["cumulative_return"].to_numpy() * 100.0).tolist(),
        colors,
        formatter=lambda value: f"{value:.1f}%",
    )
    draw_horizontal_metric_bars(
        draw,
        panel_2,
        "Sharpe Ratio",
        labels,
        metrics["sharpe_ratio"].to_numpy().tolist(),
        colors,
        formatter=lambda value: f"{value:.3f}",
    )
    draw_horizontal_metric_bars(
        draw,
        panel_3,
        "Max Drawdown",
        labels,
        (np.abs(metrics["max_drawdown"].to_numpy()) * 100.0).tolist(),
        colors,
        formatter=lambda value: f"-{value:.1f}%",
    )
    draw_footer(draw, window_text, image.height)

    output_path = FIGURES_DIR / "figure_04_shared_window_strategy_metrics.png"
    image.save(output_path)
    return output_path


def main() -> None:
    ensure_figure_dir()
    outputs = [
        build_persistence_figure(),
        build_nav_figure(),
        build_probability_figure(),
        build_metric_figure(),
    ]
    print("Final figures generated:")
    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
