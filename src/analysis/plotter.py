from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_pity_series(path: str) -> np.ndarray:
    pities = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pities.append(int(row["pity"]))
    return np.array(pities, dtype=int)


def plot_cdf(data: np.ndarray, out_path: Path, title: str) -> None:
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Pity (pulls since last 5*)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pdf(data: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    bins = max(10, int(np.sqrt(len(data))))
    plt.hist(data, bins=bins, density=True, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pity (pulls since last 5*)")
    plt.ylabel("PDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PDF/CDF of pity distribution.")
    parser.add_argument("--input", required=True, help="Path to raw CSV log")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pities = load_pity_series(args.input)
    plot_pdf(pities, out_dir / "pity_pdf.png", "Pity Distribution (PDF)")
    plot_cdf(pities, out_dir / "pity_cdf.png", "Pity Distribution (CDF)")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
