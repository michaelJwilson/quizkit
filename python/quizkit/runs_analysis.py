import json
import itertools
import polars as pl
from rich.pretty import pprint
from pathlib import Path
from typing import NamedTuple, Tuple
from quizkit.configs import TrapConfig, TrapConfigs
from quizkit.run_phase_retrieval import PerformanceMetrics

pl.Config.set_tbl_cols(-1)


class Job(NamedTuple):
    method: str
    solver_backend: str
    num_random_seeds: int
    downsample_factor: int
    smooth_phase: bool
    trap_config: TrapConfig
    initial_epsilon: float

def construct_jobs() -> Tuple[Job, ...]:
    methods = ("GD", "GS")
    smooth_phases = (False, True)
    downsample_factors = (1, 4)
    initial_epsilons = (0.0, 0.05, 0.1)
    num_random_seeds = 10
    solver_backends = ("jax",)

    trap_configs = (TrapConfigs.ON_AXIS.to_config(), TrapConfigs.OFF_AXIS.to_config())
    jobs = []

    for (
        method, backend, ds_factor, smooth, trap_conf, initial_epsilon,
    ) in itertools.product(
        methods, solver_backends, downsample_factors, smooth_phases, trap_configs, initial_epsilons,
    ):
        # NB downsampling applies to GD only
        if method != "GD" and ds_factor > 1:
            continue

        jobs.append(
            Job(
                method=method,
                solver_backend=backend,
                num_random_seeds=num_random_seeds,
                downsample_factor=ds_factor,
                smooth_phase=smooth,
                initial_epsilon=initial_epsilon,
                trap_config=trap_conf,
            )
        )

    return tuple(jobs)


def load_data(results_dir: str = "./results") -> pl.DataFrame:
    records = []

    for metric_file in Path(results_dir).rglob("performance_metrics.json"):
        solver_dir = metric_file.parent
        run_dir = solver_dir.parent.parent

        try:
            with open(metric_file) as f: metrics = json.load(f)
            with open(solver_dir / "solver_config.json") as f: solver_cfg = json.load(f)
            with open(run_dir / "run_config.json") as f: run_cfg = json.load(f)

            trap_cfg = run_cfg.pop("trap_config", {})

            record = {
                "job": {
                    **solver_cfg,
                    "trap_type": trap_cfg.get("trap_type", "unknown"),
                    "trap_id": trap_cfg.get("trap_id", -1),
                },
                "metrics": metrics,
                # Safely extract hashes for upstream aggregation
                "run_hash": str(run_cfg.get("run_hash", run_cfg.get("hash", "N/A"))),
                "solver_hash": str(solver_cfg.get("solver_hash", solver_cfg.get("hash", "N/A"))),
                "run_info": run_cfg,
            }
            records.append(record)
        except Exception as e:
            print(f"Failed to parse run at {solver_dir}: {e}")

    df = pl.DataFrame(records)

    if df.is_empty():
        return df

    df = df.with_columns(pl.col("job").hash().rank("dense").alias("job_id"))
    df = df.select(["job_id", "job", "metrics", "run_hash", "solver_hash", "run_info"])

    return df


def generate_comparison_table(
    df: pl.DataFrame,
    metric_cols: list[str],
    trap_type: str,
    epsilon: float,
    filename: str,
    caption: str,
    label: str,
    comparison_mode: str = "method",
    ds_factor: int = 1,
) -> None:
    """Filters the dataframe and writes a specialized LaTeX table dynamically based on mode."""
    
    if comparison_mode == "method":
        subset = df.filter(
            (pl.col("downsample_factor") == ds_factor) &
            (pl.col("trap_type") == trap_type) &
            (pl.col("initial_epsilon") == epsilon)
        )
        
        def get_key(row):
            m, s = row["method"], row["smooth_phase"]
            if m == "GD" and not s: return "GD"
            if m == "GS" and not s: return "GS"
            if m == "GD" and s: return "GD-$\\mathcal{C}(\\phi)$"
            if m == "GS" and s: return "GS-$\\mathcal{C}(\\phi)$"
            return None
            
        desired_keys = ["GD", "GS", "GD-$\\mathcal{C}(\\phi)$", "GS-$\\mathcal{C}(\\phi)$"]
        
    elif comparison_mode == "downsample":
        subset = df.filter(
            (pl.col("method") == "GD") &
            (pl.col("trap_type") == trap_type) &
            (pl.col("initial_epsilon") == epsilon) &
            (pl.col("downsample_factor").is_in([1, 4]))
        )
        
        def get_key(row):
            ds, s = row["downsample_factor"], row["smooth_phase"]
            if ds == 1 and not s: return "GD"
            if ds == 4 and not s: return "GD 1/4"
            if ds == 1 and s: return "GD-$\\mathcal{C}(\\phi)$"
            if ds == 4 and s: return "GD-$\\mathcal{C}(\\phi)$ 1/4"
            return None
            
        desired_keys = ["GD", "GD 1/4", "GD-$\\mathcal{C}(\\phi)$", "GD-$\\mathcal{C}(\\phi)$ 1/4"]
    else:
        raise ValueError(f"Unknown comparison_mode {comparison_mode}")

    metrics_by_run = {}

    for row in subset.iter_rows(named=True):
        run_key = get_key(row)
        if not run_key:
            continue
        
        run_metrics = {}
        for m in metric_cols:
            if m in row:
                run_metrics[m] = row[m]

            ferr_key = f"{m}_ferr"
            if ferr_key in row:
                run_metrics[ferr_key] = row[ferr_key]
        
        # Inject hashes into the dictionary so write_tex_table renders them as rows
        for hash_type in ["run_hash", "solver_hash"]:
            if hash_type in row and row[hash_type] != "N/A":
                # Truncate to standard 8-char short hash and escape any underscores for LaTeX
                val = str(row[hash_type]).replace("_", "\\_")
                run_metrics[hash_type] = val[:8] if len(val) >= 8 else val
                 
        metrics_by_run[run_key] = run_metrics
    
    # Map cleanly guaranteeing the exact column order requested
    formatted_metrics_by_run = {
        k: metrics_by_run.get(k, {}) for k in desired_keys
    }

    out_dir = Path("./runs_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    PerformanceMetrics.write_tex_table(
        filepath=out_dir / filename,
        metrics_by_run=formatted_metrics_by_run,
        caption=caption,
        label=label,
        drop_values=False,
        drop_description=True,
    )

table_variants = [
    {
        "comparison_mode": "method",
        "trap_type": "on_axis", 
        "ds_factor": 1, 
        "epsilon": 0.00,
        "filename": "first_question.tex",
        "caption": r"GD vs GS w/o $\mathcal{C}(\phi)$",
        "label": "tab:gd_vs_gs_on_axis_ds1_eps00"
    },
    {
        "comparison_mode": "method",
        "trap_type": "off_axis", 
        "ds_factor": 1, 
        "epsilon": 0.00,
        "filename": "second_question.tex",
        "caption": r"GD vs GS w/o $\mathcal{C}(\phi)$: off-axis",
        "label": "tab:gd_vs_gs_off_axis_ds1_eps00"
    },
    {
        "comparison_mode": "method",
        "trap_type": "on_axis", 
        "ds_factor": 1, 
        "epsilon": 0.05,
        "filename": "third_question.tex",
        "caption": r"GD vs GS w/o $\mathcal{C}(\phi)$: $\epsilon$-greedy",
        "label": "tab:gd_vs_gs_on_axis_ds1_eps05"
    },
    {
        "comparison_mode": "downsample",
        "trap_type": "on_axis", 
        "ds_factor": 1, # Passed but overridden by 'downsample' mode logic picking 1 & 4
        "epsilon": 0.00,
        "filename": "fourth_question.tex",
        "caption": r"GD vs GD 1/4 w/o $\mathcal{C}(\phi)$: band-limited reconstrucion",
        "label": "tab:gd_vs_gd_on_axis_ds4_eps00"
    },
]


if __name__ == "__main__":
    root ="./results_073026" 
    target_metric = "uniformity"

    jobs = construct_jobs()
    data = load_data(f"./{root}")

    if not data.is_empty():
        core = (
            data.drop("run_info")
            .unnest("job", "metrics")
            .drop(
                [
                    "trap_powers", "solver_backend", "ghost_to_trap_med_ratio",
                    "solver_runtime", "learning_rate", "maxiter", "aa_alpha",
                    "hio_beta", "anneal_rate", "trap_med", "trap_mean",
                    "trap_std", "trap_min", "trap_max", "loss_norm", "timestamp",
                ]
            )
        )

        core = core.sort(
            [
                "method", "downsample_factor", "smooth_phase",
                "trap_type", "initial_epsilon", "random_seed",
            ]
        )

        core = core.with_columns(
            pl.struct(
                [
                    "method", "downsample_factor", "smooth_phase",
                    "trap_type", "initial_epsilon",
                ]
            ).rle_id().alias("job_id")
        )

        core = core.with_columns(pl.col("random_seed").cast(pl.Int64))
        core = core.sort(["job_id", "random_seed"])
        core = core.with_row_index("index")
        core = core.select(["index", "job_id", pl.all().exclude("index", "job_id")])

        metric_cols = [
            "uniformity", "entropy", "efficiency",
            "efficiency_diffuse", "efficiency_perimeter", "pearson",
        ]

        desired_config_cols = list(Job._fields) + ["job_id", "trap_type"]
        config_cols = [c for c in desired_config_cols if c in core.columns]

        reduced_core = (
            core.group_by(config_cols)
            .agg(
                pl.col("random_seed").sort_by(target_metric, descending=True).first().alias("best_seed"),
                
                # Fetch the hashes associated with the best seed
                pl.col("run_hash").sort_by(target_metric, descending=True).first().alias("run_hash"),
                pl.col("solver_hash").sort_by(target_metric, descending=True).first().alias("solver_hash"),
                
                *[
                    pl.col(m).sort_by(target_metric, descending=True).first().alias(f"{m}")
                    for m in metric_cols
                ],
                *[
                    (pl.col(m).std() / pl.col(m).mean()).alias(f"{m}_ferr") 
                    for m in metric_cols
                ],
            )
            .sort("job_id")
        )

        reduced_core = reduced_core.select(["job_id", pl.all().exclude("job_id")])
        assert len(reduced_core) == 36

        for variant in table_variants:
            generate_comparison_table(
                df=reduced_core,
                metric_cols=metric_cols,
                **variant
            )

        question_comments = [
            "% 1) GD better than GS with/without smoothing @ on-axis, downsample_factor=1, initial_epsilon=0.00",
            "% 2) GD better than GS with/without smoothing @ off-axis, downsample_factor=1, initial_epsilon=0.00",
            "% 3) GD better than GS with/without smoothing @ on-axis, downsample_factor=1, initial_epsilon=0.05",
            "% 4) GD sampled better than GD with/without smoothing @ on-axis, initial_epsilon=0.00"
        ]

        # TODO
        merged_tex_lines = []
        out_dir = Path("./runs_analysis")

        for variant, comment in zip(table_variants, question_comments):
            table_path = out_dir / variant["filename"]
            
            with open(table_path, "r") as f:
                table_content = f.read()
                
            merged_tex_lines.append(comment)
            merged_tex_lines.append(table_content)
            merged_tex_lines.append("\n\\vspace{2em}\n")

        merged_path = out_dir / "questions.tex"
        with open(merged_path, "w") as f:
            f.write("\n".join(merged_tex_lines))
            
        print(f"Merged tables written to {merged_path}")