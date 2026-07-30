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


# TODO
#
# job
#     job_id, method, smooth_phase, downsample_factor, random_seed, initial_epsilon, trap_type,
#
# metrics:
#     uniformity, entropy, efficiency, efficiency_diffuse, efficiency_perimeter, pearson,
#     psf_wx, psf_wy, runtime
#
# to answer:
#     with metric for best seed in job & metric errors from std. across random seeds.  assumes metrics bound by (0, 1)
#         1) GD better than GS @ downsample_factor=1, on-axis @ initial_epsilon=0.0, with/without smoothing
#         2) on-axis / off-axis
#         3) on-axis with downsampling
#         4) on-axis with phase dropout (initial_epsilon==0.05) better than without.
#
#
def construct_jobs() -> Tuple[Job, ...]:
    methods = ("GD", "GS")
    smooth_phases = (False, True)
    downsample_factors = (1, 4)
    initial_epsilons = (0.0, 0.05, 0.1)
    num_random_seeds = 10
    solver_backends = ("jax",)

    # NB (float, float) or None; shift from zeroth order in the far-field basis. If None, defaults to the zeroth order position.
    #    see https://github.com/holodyne/slmsuite/blob/39243f081de020ad3ba74e672d126694b80778d2/slmsuite/holography/algorithms/_spots.py#L1423
    #
    # `"knm"``, this is ``(shape[1], shape[0])/2``.
    # ``"kxy"``, this is ``(0,0)``.
    # ``"ij"``, this is the pixel position of the zeroth order on the camera (via Fourier calibration).
    trap_configs = (TrapConfigs.ON_AXIS.to_config(), TrapConfigs.OFF_AXIS.to_config())

    jobs = []

    for (
        method,
        backend,
        ds_factor,
        smooth,
        trap_conf,
        initial_epsilon,
    ) in itertools.product(
        methods,
        solver_backends,
        downsample_factors,
        smooth_phases,
        trap_configs,
        initial_epsilons,
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
            with open(metric_file) as f:
                metrics = json.load(f)
            with open(solver_dir / "solver_config.json") as f:
                solver_cfg = json.load(f)
            with open(run_dir / "run_config.json") as f:
                run_cfg = json.load(f)

            trap_cfg = run_cfg.pop("trap_config", {})

            # Group the JSON dictionaries logically rather than flattening them
            record = {
                "job": {
                    **solver_cfg,
                    "trap_type": trap_cfg.get("trap_type", "unknown"),
                    "trap_id": trap_cfg.get("trap_id", -1),
                },
                "metrics": metrics,
                "run_info": run_cfg,
            }
            records.append(record)
        except Exception as e:
            print(f"Failed to parse run at {solver_dir}: {e}")

    df = pl.DataFrame(records)

    if df.is_empty():
        return df

    # Dynamically generate a job_id by hashing the 'job' struct.
    # All rows with identical solver configurations will share the same job_id.
    df = df.with_columns(pl.col("job").hash().rank("dense").alias("job_id"))

    # Reorder top-level structural columns
    df = df.select(["job_id", "job", "metrics", "run_info"])

    return df


if __name__ == "__main__":
    target_metric = "uniformity"

    jobs = construct_jobs()
    data = load_data()

    if not data.is_empty():
        # NB job_id (i64), job (struct), metrics (struct), run_info (struct)
        # print(data)

        core = (
            data.drop("run_info")
            .unnest("job", "metrics")
            .drop(
                [
                    "trap_powers",
                    "solver_backend",
                    "ghost_to_trap_med_ratio",
                    "solver_runtime",
                    "learning_rate",
                    "maxiter",
                    "aa_alpha",
                    "hio_beta",
                    "anneal_rate",
                    "trap_med",
                    "trap_mean",
                    "trap_std",
                    "trap_min",
                    "trap_max",
                    "loss_norm",
                    "timestamp",
                ]
            )
        )

        core = core.sort(
            [
                "method",
                "downsample_factor",
                "smooth_phase",
                "trap_type",
                "initial_epsilon",
                "random_seed",
            ]
        )

        # NB rebuild job id
        core = core.with_columns(
            pl.struct(
                [
                    "method",
                    "downsample_factor",
                    "smooth_phase",
                    "trap_type",
                    "initial_epsilon",
                ]
            )
            .rle_id()
            .alias("job_id")
        )

        # TODO HACK random_seed as int upstream.
        core = core.with_columns(pl.col("random_seed").cast(pl.Int64))
        core = core.sort(["job_id", "random_seed"])
        core = core.sort(["job_id"])

        core = core.with_row_index("index")
        core = core.select(["index", "job_id", pl.all().exclude("index", "job_id")])

        # pprint(core)

        first_job = core.filter(pl.col("job_id") == 0)

        # print(first_job)

        # TODO HARDCODE
        metric_cols = [
            "uniformity",
            "entropy",
            "efficiency",
            "efficiency_diffuse",
            "efficiency_perimeter",
            "pearson",
        ]

        desired_config_cols = list(Job._fields) + ["job_id", "trap_type"]
        config_cols = [c for c in desired_config_cols if c in core.columns]

        reduced_core = (
            core.group_by(config_cols)
            .agg(
                pl.col("random_seed")
                .sort_by(target_metric, descending=True)
                .first()
                .alias("best_seed"),
                
                # 1. Keep the metrics for the best seed
                *[
                    pl.col(m)
                    .sort_by(target_metric, descending=True)
                    .first()
                    .alias(f"{m}")
                    for m in metric_cols
                ],
                
                # 2. Calculate fractional error in percent: (std / mean)
                *[
                    (pl.col(m).std() / pl.col(m).mean()).alias(f"{m}_ferr") 
                    for m in metric_cols
                ],
            )
            .sort("job_id")
        )

        reduced_core = reduced_core.select(["job_id", pl.all().exclude("job_id")])

        assert len(reduced_core) == 36

        # pprint(reduced_core)

        first_question = reduced_core.filter(
            (pl.col("downsample_factor") == 1) &
            (pl.col("trap_type") == "on_axis") &
            (pl.col("initial_epsilon") == 0.0)
        )

        pprint(first_question)

        def get_run_key(method_str: str, is_smooth: bool) -> str:
            if method_str == "GD" and not is_smooth:
                return "GD"
            elif method_str == "GS" and not is_smooth:
                return "GS"
            elif method_str == "GD" and is_smooth:
                return "GD-SMOOTH"
            elif method_str == "GS" and is_smooth:
                return "GS-SMOOTH"
            return "UNKNOWN"

        metrics_by_run = {}

        for row in first_question.iter_rows(named=True):
            method = row["method"]
            smooth = row["smooth_phase"]
            
            run_key = get_run_key(method, smooth)
            
            run_metrics = {}
            for m in metric_cols:
                run_metrics[m] = row.get(m)

                ferr_key = f"{m}_ferr"
                if ferr_key in row:
                     run_metrics[ferr_key] = row.get(ferr_key)
                     
            metrics_by_run[run_key] = run_metrics
        
        formatted_metrics_by_run = {
            "GD": metrics_by_run["GD"],
            "GS": metrics_by_run["GS"],
            "GD-$\\mathcal{C}(\\phi)$": metrics_by_run["GD-SMOOTH"],
            "GS-$\\mathcal{C}(\\phi)$": metrics_by_run["GS-SMOOTH"],
        }

        out_dir = Path("./runs_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        PerformanceMetrics.write_tex_table(
            filepath=out_dir / "first_question.tex",
            metrics_by_run=formatted_metrics_by_run,
            # caption="Comparison of Gradient Descent (GD) and Gerchberg-Saxton (GS) with and without phase smoothing $\\mathcal{C}(\\phi)$. Evaluated on-axis with downsample factor 1 and zero initial epsilon.",
            label="tab:gd_vs_gs_smooth",
            drop_values=True,
        )
