from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("epidemic_forecasting")
OUTDIR = ROOT / "results" / "first_valid_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "name": "COVID UK",
        "pred_path": ROOT / "results" / "covid19" / "uk_14days" / "uk_ttt_7b_100" / "job_5989922" / "step_0001" / "parent_01_rollout_02" / "predictions.npy",
        "ref_csv": ROOT / "results" / "covid19" / "final_100" / "uk_ttt" / "best_forecast_vs_actual.csv",
        "title": "COVID-19 UK — TTT First Valid Forecast (14 days)",
        "output": "covid_uk_ttt_first_valid.png",
        "series_names": [
            "England",
            "Scotland",
            "Wales",
        ],
        "max_panels": 3,
    },
    {
        "name": "COVID US",
        "pred_path": ROOT / "results" / "covid19" / "us_14days" / "ttt" / "job_5989964" / "step_0001" / "parent_01_rollout_02" / "predictions.npy",
        "ref_csv": ROOT / "results" / "covid19" / "final_100" / "us_ttt" / "best_forecast_vs_actual.csv",
        "title": "COVID-19 US — TTT First Valid Forecast (14 days)",
        "output": "covid_us_ttt_first_valid.png",
        "series_names": [
            "California",
            "Florida",
            "New York",
            "Texas",
        ],
        "max_panels": 4,
    },
    {
        "name": "Dengue Colombia",
        "pred_path": ROOT / "results" / "dengue" / "colombia_8weeks" / "colombia_ttt_7b_100" / "job_6017916" / "step_0001" / "parent_01_rollout_02" / "predictions.npy",
        "ref_csv": ROOT / "results" / "dengue" / "colombia_8weeks" / "colombia_ttt_7b_100" / "job_6017916" / "best_forecast_vs_actual.csv",
        "title": "Dengue Colombia — TTT First Valid Forecast (8 weeks)",
        "output": "dengue_colombia_ttt_first_valid.png",
        "series_names": None,
        "max_panels": 4,
    },
    {
        "name": "Dengue Panama",
        "pred_path": ROOT / "results" / "dengue" / "panama_8weeks" / "panama_ttt_7b_100" / "job_6017918" / "step_0001" / "parent_01_rollout_04" / "predictions.npy",
        "ref_csv": ROOT / "results" / "dengue" / "panama_8weeks" / "panama_ttt_7b_100" / "job_6017918" / "best_forecast_vs_actual.csv",
        "title": "Dengue Panama — TTT First Valid Forecast (8 weeks)",
        "output": "dengue_panama_ttt_first_valid.png",
        "series_names": None,
        "max_panels": 4,
    },
]

def choose_series(ref_df, series_names, max_panels):
    meta = (
        ref_df[["series_index", "series_name"]]
        .drop_duplicates()
        .sort_values("series_index")
        .reset_index(drop=True)
    )

    if series_names is not None:
        rows = []

        for name in series_names:
            match = meta[
                meta["series_name"].str.replace("_", " ", regex=False)
                == name
            ]

            if match.empty:
                raise ValueError(
                    f"Could not find requested series: {name}"
                )

            rows.append(match.iloc[0])

        return pd.DataFrame(rows).reset_index(drop=True)

    return meta.head(max_panels)

def plot_run(run):
    pred = np.load(run["pred_path"])
    ref_df = pd.read_csv(run["ref_csv"])

    chosen = choose_series(
        ref_df,
        run["series_names"],
        run["max_panels"],
    )

    n = len(chosen)

    if n == 3:
        nrows, ncols = 1, 3
        figsize = (18, 5)
    elif n <= 2:
        nrows, ncols = 1, n
        figsize = (14, 5)
    else:
        nrows, ncols = 2, 2
        figsize = (16, 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        series_idx = int(row["series_index"])
        series_name = row["series_name"]

        part = (
            ref_df[ref_df["series_name"] == series_name]
            .copy()
            .sort_values("day")
        )

        actual = part["actual"].to_numpy()
        predicted = pred[:, series_idx]

        x = np.arange(len(actual))
        width = 0.38

        ax.bar(x - width / 2, predicted, width, label="Predicted")
        ax.bar(x + width / 2, actual, width, label="Actual")

        ax.set_title(series_name.replace("_", " "))
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Cases")
        ax.set_xticks(x)
        ax.set_xticklabels(part["day"].astype(str))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(run["title"], fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = OUTDIR / run["output"]
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {run['name']}")
    print(f"Saved to: {outpath}")

def main():
    for run in RUNS:
        print("=" * 70)
        print(run["name"])
        print("Predictions:", run["pred_path"])
        print("Reference CSV:", run["ref_csv"])
        plot_run(run)

    print("=" * 70)
    print("All first-valid figures generated.")
    print(f"Output directory: {OUTDIR}")

if __name__ == "__main__":
    main()
