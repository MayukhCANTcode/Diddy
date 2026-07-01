"""
=========================================================
File Name : create_statistics.py

Purpose:
    Generate visualization charts for deforestation
    statistics.

Author : Mayukh Das
=========================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT = PROJECT_ROOT / "outputs"

# -----------------------------------------
# Read CSV files
# -----------------------------------------

statistics = pd.read_csv(
    OUTPUT / "change_statistics.csv"
)

transitions = pd.read_csv(
    OUTPUT / "transition_matrix.csv"
)

# -----------------------------------------
# Forest Loss Pie Chart
# -----------------------------------------

forest_total = int(
    statistics.loc[
        statistics["Metric"] == "Forest Patches (2018)",
        "Value"
    ].values[0]
)

forest_loss = int(
    statistics.loc[
        statistics["Metric"] == "Forest Loss Patches",
        "Value"
    ].values[0]
)

forest_remaining = forest_total - forest_loss

plt.figure(figsize=(6, 6))

plt.pie(

    [forest_remaining, forest_loss],

    labels=[

        "Remaining Forest",

        "Forest Loss"

    ],

    autopct="%1.1f%%",

    startangle=90

)

plt.title("Forest Loss (2018-2024)")

plt.savefig(
    OUTPUT / "forest_loss_pie.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------
# Transition Bar Chart
# -----------------------------------------

transition_labels = (

    transitions["To"]

)

transition_counts = (

    transitions["Patch Count"]

)

plt.figure(figsize=(10, 6))

plt.bar(

    transition_labels,

    transition_counts

)

plt.title("Forest Transition Types")

plt.xlabel("Converted To")

plt.ylabel("Patch Count")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(

    OUTPUT /

    "transition_bar_chart.png",

    dpi=300

)

plt.close()

print("="*60)
print("STATISTICS CHARTS GENERATED")
print("="*60)
print("forest_loss_pie.png")
print("transition_bar_chart.png")
