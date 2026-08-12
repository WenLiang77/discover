from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "epidemic_forecasting/results/covid19/final_100"
)

OUTPUT = Path(
    "epidemic_forecasting/results/covid19/final_100/figures"
)
OUTPUT.mkdir(parents=True, exist_ok=True)


def plot_four_series(
    csv_path,
    series_names,
    title,
    output_name,
):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    axes = axes.flatten()

    for ax, series_name in zip(
        axes,
        series_names,
    ):
        part = df[
            df["series_name"] == series_name
        ].copy()

        x = np.arange(len(part))
        width = 0.38

        ax.bar(
            x - width / 2,
            part["predicted"],
            width,
            label="Predicted",
        )

        ax.bar(
            x + width / 2,
            part["actual"],
            width,
            label="Actual",
        )

        ax.set_title(
            series_name.replace("_", " ")
        )

        ax.set_xlabel("Forecast day")
        ax.set_ylabel("Cases")

        ax.set_xticks(x)
        ax.set_xticklabels(
            part["day"].astype(str),
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.legend()

    fig.suptitle(
        title,
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path = OUTPUT / output_name

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# UK
# ============================================================

uk_series = [
    "England",
    "Northern Ireland",
    "Scotland",
    "Wales",
]

plot_four_series(
    ROOT
    / "uk_no_ttt"
    / "best_forecast_vs_actual.csv",
    uk_series,
    "UK COVID-19 — No-TTT Best Forecast (14 days)",
    "uk_no_ttt_forecast.png",
)

plot_four_series(
    ROOT
    / "uk_ttt"
    / "best_forecast_vs_actual.csv",
    uk_series,
    "UK COVID-19 — TTT Best Forecast (14 days)",
    "uk_ttt_forecast.png",
)


# ============================================================
# US
# ============================================================

us_series = [
    "California",
    "Florida",
    "New_York",
    "Texas",
]

plot_four_series(
    ROOT
    / "us_no_ttt"
    / "best_forecast_vs_actual.csv",
    us_series,
    "US COVID-19 — No-TTT Best Forecast (14 days)",
    "us_no_ttt_forecast.png",
)

plot_four_series(
    ROOT
    / "us_ttt"
    / "best_forecast_vs_actual.csv",
    us_series,
    "US COVID-19 — TTT Best Forecast (14 days)",
    "us_ttt_forecast.png",
)
