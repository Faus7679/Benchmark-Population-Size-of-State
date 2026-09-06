import csv
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "NST-EST2024-ALLDATA.csv"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_state_row(row):
    if row["SUMLEV"] != "040":
        return False
    name = row["NAME"]
    non_states = {"District of Columbia", "Puerto Rico"}
    return name not in non_states


def linear_regression(xs, ys):
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    y_hat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1 - (ss_res / ss_tot)
    return slope, intercept, r2


def predict(slope, intercept, year):
    return intercept + slope * year


def main():
    rows = load_rows(DATA_PATH)
    states = [r for r in rows if is_state_row(r)]

    years = [2020, 2021, 2022, 2023, 2024]
    md = next(r for r in states if r["NAME"] == "Maryland")
    md_pops = [int(md[f"POPESTIMATE{y}"]) for y in years]

    slope, intercept, r2 = linear_regression(years, md_pops)

    # Ten years from current assignment year (2026) -> 2036
    target_year = 2036
    md_pred_2036 = round(predict(slope, intercept, target_year))

    growth = []
    for r in states:
        p0 = int(r["POPESTIMATE2020"])
        p4 = int(r["POPESTIMATE2024"])
        rate_pct = ((p4 - p0) / p0) * 100
        growth.append((r["NAME"], rate_pct))
    growth.sort(key=lambda t: t[1], reverse=True)
    md_rank = next(i + 1 for i, t in enumerate(growth) if t[0] == "Maryland")
    md_growth_pct = next(t[1] for t in growth if t[0] == "Maryland")

    # Figure 1: model and forecast
    x_plot = years + [target_year]
    y_fit = [predict(slope, intercept, x) for x in x_plot]

    plt.figure(figsize=(9, 5))
    plt.plot(years, md_pops, "o", label="Observed Maryland population")
    plt.plot(x_plot, y_fit, "-", label="Linear regression fit")
    plt.plot([target_year], [md_pred_2036], "r*", markersize=14, label=f"Predicted {target_year}")
    plt.title("Maryland Population Dynamics (Census 2020-2024)")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "maryland_regression_forecast.png", dpi=180)
    plt.close()

    # Figure 2: screenshot-like results panel
    summary = (
        "Maryland Population Regression Results\n"
        f"Years used: {years[0]}-{years[-1]}\n"
        f"Slope (people/year): {slope:,.0f}\n"
        f"Intercept: {intercept:,.0f}\n"
        f"R^2: {r2:.4f}\n"
        f"Maryland growth 2020-2024: {md_growth_pct:.4f}%\n"
        f"Maryland growth-rank among 50 states: {md_rank}\n"
        f"Predicted population in {target_year}: {md_pred_2036:,}"
    )

    plt.figure(figsize=(10, 4), facecolor="#1e1e1e")
    ax = plt.gca()
    ax.set_facecolor("#1e1e1e")
    ax.text(0.02, 0.95, summary, va="top", ha="left", family="monospace", fontsize=11, color="#d4d4d4")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "maryland_results_screenshot.png", dpi=180)
    plt.close()

    print("Maryland model summary")
    print(summary)


if __name__ == "__main__":
    main()
