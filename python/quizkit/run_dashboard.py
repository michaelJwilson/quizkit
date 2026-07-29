import json
from pathlib import Path

import panel as pn
import polars as pl
import hvplot
import hvplot.polars

import panel as pn

# NB inject modern web fonts and smooth card styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

body, .bk, .bk-root, .panel-title {
    font-family: 'Inter', sans-serif !important;
}

/* Remove harsh borders and add subtle floating shadows to cards */
.bk-panel-models-layout-Card {
    border: 1px solid #374151 !important; /* Tailwind Gray 700 */
    border-radius: 8px !important;
    background-color: #1F2937 !important; /* Tailwind Gray 800 */
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}
"""

pn.extension("tabulator", raw_css=[custom_css], sizing_mode="stretch_width")

"""
panel serve python/quizkit/run_dashboard.py --show
"""


def load_data(results_dir: str = "./results") -> pl.DataFrame:
    records = []

    # NB find all metric files (which indicate a completed solver run)
    for metric_file in Path(results_dir).rglob("performance_metrics.json"):
        solver_dir = metric_file.parent
        run_dir = solver_dir.parent.parent

        try:
            with open(metric_file) as f:
                metrics = json.load(f)
            with open(solver_dir / "solver_config.json") as f:
                solver_cfg = json.load(f)
            with open(run_dir / "run_config.json") as f:
                run_cfg = json.load(f)

            metrics.pop("trap_powers", None)
            trap_cfg = run_cfg.pop("trap_config", {})
            record = {
                **metrics,
                **solver_cfg,
                **run_cfg,
                "trap_type": trap_cfg.get("trap_type", "unknown"),
                "trap_id": trap_cfg.get("trap_id", -1),
            }
            records.append(record)
        except Exception as e:
            print(f"Failed to parse run at {solver_dir}: {e}")

    df = pl.DataFrame(records)

    if not df.is_empty():
        df = df.with_columns(
            [
                pl.col("method").cast(pl.String).cast(pl.Categorical),
                pl.col("trap_type").cast(pl.String).cast(pl.Categorical),
                pl.col("random_seed").cast(pl.String).cast(pl.Categorical),
            ]
        )

        ordered_cols = [
            # Identifiers
            "hash",
            "timestamp",
            # Trap Config
            "trap_id",
            "trap_type",
            # "array_shape",
            # "array_pitch",
            # "array_center",
            # Run Config
            # "wavelength",
            # "pixel_pitch",
            # "slm_shape",
            # "comment",
            # Solver Config
            "method",
            "maxiter",
            "solver_backend",
            "solver_runtime",
            "smooth_phase",
            # "smooth_sigma",
            # "loss_norm",
            # "learning_rate",
            "random_seed",
            # "initial_epsilon",
            # "anneal_rate",
            # Metrics
            "efficiency",
            # "stray_light_fraction",
            "pearson",
            # "trap_cv",
            # "trap_mean",
            # "trap_min",
            # "trap_max",
            "trap_uniformity_minmax",
            # "ghost_to_mean_ratio",
            # "ghost_to_dimmest_ratio",
            # "signal_to_background_floor",
        ]

        final_cols = [c for c in ordered_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in final_cols]

        df = df.select(final_cols + extra_cols)

    return df


df = load_data()

if df.is_empty():
    pn.pane.Markdown("## No data found in `./results`!").servable()
else:
    # --- Widgets ---
    metric_cols = [
        "efficiency",
        "pearson",
        "trap_cv",
        "uniformity",
        "ghost_to_mean_ratio",
    ]
    cat_cols = ["method", "trap_type", "random_seed"]

    w_x_axis = pn.widgets.Select(
        name="X-Axis (Scatter)", options=metric_cols, value="pearson"
    )
    w_y_axis = pn.widgets.Select(
        name="Y-Axis (Scatter)", options=metric_cols, value="trap_cv"
    )
    w_color_by = pn.widgets.Select(name="Color By", options=cat_cols, value="method")

    w_trap_filter = pn.widgets.MultiChoice(
        name="Filter: Trap Type",
        options=df["trap_type"].unique().to_list(),
        value=df["trap_type"].unique().to_list(),
    )

    w_method_filter = pn.widgets.MultiChoice(
        name="Filter: Solver Method",
        options=df["method"].unique().to_list(),
        value=df["method"].unique().to_list(),
    )

    # --- Helper ---
    def get_filtered_df(traps, methods):
        return df.filter(
            pl.col("trap_type").is_in(traps) & pl.col("method").is_in(methods)
        )

    # --- Reactive KPIs (Top Row Indicators) ---
    @pn.depends(w_trap_filter, w_method_filter)
    def kpi_indicators(traps, methods):
        filtered = get_filtered_df(traps, methods)
        if filtered.is_empty():
            return pn.Row(pn.pane.Markdown("No data"))

        # Calculate dynamic top-line metrics
        best_uniformity = filtered["trap_uniformity_minmax"].max()
        best_eff = filtered["efficiency"].max()
        run_count = len(filtered)

        return pn.Row(
            pn.indicators.Number(
                name="Total runs",
                value=run_count,
                format="{value}",
                colors=[(1, "var(--neutral-fill-active)")],
                font_size="32pt",
                title_size="14pt",  # <-- Add these
            ),
            pn.indicators.Number(
                name="Best efficiency",
                value=best_eff,
                format="{value:.4f}",
                colors=[(1, "var(--success-fill-rest)")],
                font_size="32pt",
                title_size="14pt",  # <-- Add these
            ),
            pn.indicators.Number(
                name="Best uniformity",
                value=best_uniformity,
                format="{value:.4f}",
                colors=[(1, "var(--accent-fill-rest)")],
                font_size="32pt",
                title_size="14pt",  # <-- Add these
            ),
            sizing_mode="stretch_width",
        )

    # --- Reactive Plots ---
    @pn.depends(w_x_axis, w_y_axis, w_color_by, w_trap_filter, w_method_filter)
    def plot_scatter(x, y, color_col, traps, methods):
        filtered = get_filtered_df(traps, methods)
        if filtered.is_empty():
            return pn.pane.Markdown("No data.")

        return filtered.hvplot.scatter(
            x=x,
            y=y,
            by=color_col,
            size=150,
            alpha=0.85,
            hover_cols=["hash", "random_seed"],
            cmap="Category10",  # Modern categorical colors
            line_color="white",  # White borders on scatter points
            line_width=0.5,
            grid=False,  # Remove grid for a cleaner look
            responsive=True,
            min_height=380,
        )

    @pn.depends(w_x_axis, w_color_by, w_trap_filter, w_method_filter)
    def plot_histogram(x, color_col, traps, methods):
        filtered = get_filtered_df(traps, methods)
        if filtered.is_empty():
            return pn.pane.Markdown("No data.")

        return filtered.hvplot.hist(
            y=x,
            by=color_col,
            alpha=0.7,
            bins=30,
            cmap="Category10",
            line_width=0,  # Remove bin borders
            responsive=True,
            min_height=380,
        )

    # --- Reactive Tabulator Table ---
    @pn.depends(w_trap_filter, w_method_filter)
    def filtered_table(traps, methods):
        filtered = get_filtered_df(traps, methods)
        return pn.widgets.Tabulator(
            filtered.to_pandas(),
            pagination="remote",
            page_size=10,
            theme="fast",
            sizing_mode="stretch_width",
        )

    # --- Layout Assembly ---
    sidebar = pn.Column(
        pn.pane.Markdown("## Settings", margin=(0, 0, 10, 0)),
        w_x_axis,
        w_y_axis,
        w_color_by,
        pn.layout.Divider(),
        w_trap_filter,
        w_method_filter,
    )

    # 1. Group the plots into a dynamic Tab component
    plot_tabs = pn.Tabs(
        ("Scatter Plot", plot_scatter),
        ("1D Distribution", plot_histogram),
        dynamic=True,  # Renders only the active tab to save memory
        sizing_mode="stretch_both",
    )

    # 2. Wrap the tabs in a single styling Card
    tabs_card = pn.Card(
        plot_tabs,
        title="Metric Exploration",
        margin=10,
        min_height=450,
        sizing_mode="stretch_both",
    )

    # 3. Table Card
    table_card = pn.Card(
        filtered_table, title="Run Data", margin=10, sizing_mode="stretch_width"
    )

    # 4. Main UI Column
    main_content = pn.Column(kpi_indicators, tabs_card, table_card)

    template = pn.template.FastListTemplate(
        title="Quizkit::phase retrieval",
        sidebar=[sidebar],
        main=[main_content],
        theme="dark",
        font="Inter",
        # Modern Dark Slate Theme (Tailwind Inspired)
        background_color="#111827",  # Deep gray background (not pure black)
        header_background="#1F2937",  # Lighter gray for the header
        accent_base_color="#8B5CF6",  # Vibrant violet accent for sliders/toggles
        header_color="#F9FAFB",  # Off-white header text
        sidebar_width=320,
    )

    template.servable()
