#!/usr/bin/env python3
"""
SUMO Summary Comparison Tool (Final + Waiting Time)
--------------------------------------------------
Compare three methods:

    • Fixed-Time
    • DDQN
    • LLIRL

Generates:
    - waiting_over_time.png
    - running_over_time.png
    - arrived_cumulative.png
    - meanSpeed_over_time.png
    - meanTravelTime_over_time.png
    - meanWaitingTime_over_time.png   <-- NEW

Outputs:
    - metrics_compare.csv
"""

import os
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------
# Parse summary XML
# ----------------------------------------------------

def parse_summary(path):
    """Load SUMO summary.xml into a Pandas DataFrame."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Summary file not found: {path}")

    rows = []
    root = ET.parse(path).getroot()

    for step in root.iter("step"):
        row = {}
        for k, v in step.attrib.items():
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v

        if "time" not in row:
            continue

        rows.append(row)

    if not rows:
        raise RuntimeError(f"Summary file is empty: {path}")

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    # Ensure basic columns exist
    for col in ["running", "waiting", "arrived", "departed", "inserted", "ended"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------
# Metrics
# ----------------------------------------------------

def compute_metrics(df, tag):
    m = {"run": tag}

    m["duration"] = float(df["time"].iloc[-1])

    # vehicle counts
    for col in ["waiting", "running", "arrived"]:
        m[f"{col}_mean"] = float(np.nanmean(df[col]))
        m[f"{col}_max"] = float(np.nanmax(df[col]))

    # final counts
    for col in ["arrived", "departed", "inserted", "ended"]:
        m[f"{col}_final"] = float(df[col].iloc[-1])

    # traffic quality metrics
    for col in ["meanSpeed", "meanTravelTime", "meanWaitingTime", "waitingTime"]:
        if col in df.columns:
            m[f"{col}_mean"] = float(np.nanmean(df[col]))
            m[f"{col}_max"] = float(np.nanmax(df[col]))

    return m


def cumulative_series(df, col):
    """Convert arrived -> cumulative arrived."""
    arr = df[col].to_numpy()
    inc = np.diff(np.r_[arr[:1], arr]).clip(min=0)
    out = df.copy()
    out[col + "_cum"] = np.cumsum(inc)
    return out


def plot_multi(dfs, labels, colors, col, title, out_png, ylabel=None):
    plt.figure()
    for df, lab, color in zip(dfs, labels, colors):
        if col in df.columns:
            plt.plot(df["time"], df[col], label=lab, color=color)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel if ylabel else col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fixed", required=True, help="Path to Fixed-Time summary.xml")
    ap.add_argument("--ddqn", required=True, help="Path to DDQN summary.xml")
    ap.add_argument("--llirl", required=True, help="Path to LLIRL summary.xml")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder")

    args = ap.parse_args()

    # Output folder
    outdir = args.outdir or os.path.join(
        "reports", "compare",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    ensure_dir(outdir)

    print("\n===== SUMMARY FILES =====")
    print(f"Fixed : {args.fixed}")
    print(f"DDQN  : {args.ddqn}")
    print(f"LLIRL : {args.llirl}")

    # Load data
    df_fixed = parse_summary(args.fixed)
    df_ddqn  = parse_summary(args.ddqn)
    df_llirl = parse_summary(args.llirl)

    dfs    = [df_fixed, df_ddqn, df_llirl]
    labels = ["Fixed", "DDQN", "LLIRL"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    # Save metrics
    metrics = [
        compute_metrics(df_fixed, "Fixed"),
        compute_metrics(df_ddqn, "DDQN"),
        compute_metrics(df_llirl, "LLIRL"),
    ]
    pd.DataFrame(metrics).to_csv(
        os.path.join(outdir, "metrics_compare.csv"), index=False
    )

    # ------------------------------------------------
    # Plots
    # ------------------------------------------------

    # Waiting vehicles
    plot_multi(
        dfs, labels, colors,
        "waiting",
        "Waiting vehicles over time",
        os.path.join(outdir, "waiting_over_time.png"),
        "Number of waiting vehicles"
    )

    # Running vehicles
    plot_multi(
        dfs, labels, colors,
        "running",
        "Running vehicles over time",
        os.path.join(outdir, "running_over_time.png"),
        "Number of running vehicles"
    )

    # Cumulative arrived
    plt.figure()
    for df, lab, color in zip(dfs, labels, colors):
        dfc = cumulative_series(df, "arrived")
        plt.plot(dfc["time"], dfc["arrived_cum"], label=lab, color=color)

    plt.title("Cumulative arrived vehicles")
    plt.xlabel("Time (s)")
    plt.ylabel("Arrived (cumulative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "arrived_cumulative.png"), dpi=150)
    plt.close()

    # Mean speed
    plot_multi(
        dfs, labels, colors,
        "meanSpeed",
        "Average speed (m/s) over time",
        os.path.join(outdir, "meanSpeed_over_time.png"),
        "Speed (m/s)"
    )

    # Mean travel time
    plot_multi(
        dfs, labels, colors,
        "meanTravelTime",
        "Average travel time (s) over time",
        os.path.join(outdir, "meanTravelTime_over_time.png"),
        "Travel time (s)"
    )

    # Mean waiting time (fallback supported)
    if "meanWaitingTime" in df_fixed.columns:
        col = "meanWaitingTime"
        title = "Average waiting time (s) over time"
        fname = "meanWaitingTime_over_time.png"
        ylabel = "Waiting time (s)"
    elif "waitingTime" in df_fixed.columns:
        col = "waitingTime"
        title = "Total waiting time (s) over time"
        fname = "waitingTime_over_time.png"
        ylabel = "Total waiting time (s)"
    else:
        col = None

    if col:
        plot_multi(
            dfs, labels, colors,
            col,
            title,
            os.path.join(outdir, fname),
            ylabel
        )

    print("\n[OK] All charts saved in:", outdir)


if __name__ == "__main__":
    main()
