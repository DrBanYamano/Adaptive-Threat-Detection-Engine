#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAHORAGA SENTINEL AI v2
Adaptive Threat Detection Engine

Upgrade:
✔ Korean + English UI
✔ Better CLI Dashboard
✔ CSV Log Input
✔ Real-time Detect Mode
✔ Risk Level
✔ Save JSON Report
✔ History Table
✔ Better Error Handling

Run:
python detector_v2.py
python detector_v2.py --csv traffic.csv
python detector_v2.py --json report.json
"""

import os
import sys
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# ==================================================
# CONFIG
# ==================================================
MODEL_PATH = "isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLD_PATH = "threshold.txt"

results_history = []

# ==================================================
# LOAD RESOURCES
# ==================================================
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        with open(THRESHOLD_PATH, "r") as f:
            threshold = float(f.read())

        return model, scaler, threshold

    except Exception as e:
        console.print(f"[bold red]ERROR:[/] {e}")
        sys.exit(1)

# ==================================================
# UI
# ==================================================
def banner(input_dim, threshold):
    console.print(Panel.fit(
        f"[bold cyan]MAHORAGA SENTINEL AI v2[/]\n"
        f"[yellow]Adaptive Threat Detection Engine[/]\n\n"
        f"Features: [green]{input_dim}[/] | "
        f"Threshold: [magenta]{threshold:.4f}[/]",
        border_style="cyan"
    ))

# ==================================================
# RISK LEVEL
# ==================================================
def risk_level(score, threshold):
    if score < threshold - 0.10:
        return "CRITICAL", "bold red"
    elif score < threshold:
        return "HIGH", "red"
    elif score < threshold + 0.05:
        return "MEDIUM", "yellow"
    else:
        return "LOW", "green"

# ==================================================
# DETECT SINGLE
# ==================================================
def detect_single(ip_name, values, model, scaler, threshold):

    try:
        arr = np.array([float(x) for x in values]).reshape(1, -1)

    except:
        console.print("[red]Invalid numeric input[/]")
        return

    scaled = scaler.transform(arr)
    score = model.decision_function(scaled)[0]

    abnormal = score < threshold
    status = "ANOMALY 🚨" if abnormal else "NORMAL ✅"

    risk, color = risk_level(score, threshold)

    item = {
        "time": str(datetime.now()),
        "ip": ip_name,
        "score": float(score),
        "status": status,
        "risk": risk
    }

    results_history.append(item)

    console.print(
        f"[{color}] {ip_name:<15} "
        f"Score: {score:.4f} "
        f"Status: {status:<12} "
        f"Risk: {risk} [/]"
    )

# ==================================================
# SHOW TABLE
# ==================================================
def show_history():

    if not results_history:
        console.print("[yellow]No data[/]")
        return

    table = Table(title="Detection History", show_lines=True)

    table.add_column("Time", style="cyan")
    table.add_column("IP")
    table.add_column("Score")
    table.add_column("Status")
    table.add_column("Risk")

    for r in results_history:
        table.add_row(
            r["time"][:19],
            r["ip"],
            f'{r["score"]:.4f}',
            r["status"],
            r["risk"]
        )

    console.print(table)

# ==================================================
# SAVE JSON
# ==================================================
def save_json(file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(results_history, f, indent=4, ensure_ascii=False)

    console.print(f"[green]Saved:[/] {file}")

# ==================================================
# CSV MODE
# ==================================================
def process_csv(file, model, scaler, threshold):

    try:
        df = pd.read_csv(file)

    except Exception as e:
        console.print(f"[red]CSV Error:[/] {e}")
        return

    if "ip" not in df.columns:
        console.print("[red]CSV must contain 'ip' column[/]")
        return

    feature_cols = [c for c in df.columns if c != "ip"]

    for _, row in df.iterrows():
        ip = row["ip"]
        vals = row[feature_cols].tolist()
        detect_single(ip, vals, model, scaler, threshold)

# ==================================================
# REALTIME DEMO
# ==================================================
def realtime_demo(model, scaler, threshold, dim):

    console.print("[cyan]Realtime Mode Started (Ctrl+C exit)[/]")

    try:
        while True:
            fake = np.random.normal(50, 10, dim)

            if np.random.rand() < 0.2:
                fake = fake * np.random.uniform(2, 4)

            detect_single(
                f"LiveIP_{np.random.randint(1,999)}",
                fake,
                model,
                scaler,
                threshold
            )

            time.sleep(2)

    except KeyboardInterrupt:
        console.print("[red]Realtime Stopped[/]")

# ==================================================
# MAIN
# ==================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--json")
    parser.add_argument("--live", action="store_true")

    args = parser.parse_args()

    model, scaler, threshold = load_resources()

    input_dim = scaler.n_features_in_

    banner(input_dim, threshold)

    # CSV MODE
    if args.csv:
        process_csv(args.csv, model, scaler, threshold)

        if args.json:
            save_json(args.json)

        return

    # LIVE MODE
    if args.live:
        realtime_demo(model, scaler, threshold, input_dim)
        return

    # INTERACTIVE MODE
    while True:

        cmd = console.input(
            "\n[bold cyan]Input:[/] "
            "IP,val1,val2,... / show / save / exit\n> "
        )

        if cmd.lower() == "exit":
            break

        if cmd.lower() == "show":
            show_history()
            continue

        if cmd.lower() == "save":
            save_json("report.json")
            continue

        parts = [x.strip() for x in cmd.split(",")]

        if len(parts) == input_dim + 1:
            ip = parts[0]
            vals = parts[1:]
            detect_single(ip, vals, model, scaler, threshold)
        else:
            console.print("[yellow]Wrong format[/]")

if __name__ == "__main__":
    main()