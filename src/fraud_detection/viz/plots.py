import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _pretty(col: str) -> str:
    return col.replace("_", " ").title()


# -----------------------------
# Class distribution
# -----------------------------
def plot_class_distribution(class_counts):
    labels = ["Non-Fraud", "Fraud"]
    values = class_counts.reindex([0, 1], fill_value=0).values
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Distribution (Count)")

    for i, v in enumerate(values):
        axes[0].text(i, v * 1.01, f"{int(v):,}",
                     ha="center", fontweight="bold")

    # Pie
    axes[1].pie(
        values,
        labels=labels,
        autopct="%1.2f%%",
        colors=colors,
        explode=[0, 0.1],
        startangle=90,
    )
    axes[1].set_title("Class Distribution (Percentage)")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Numeric distribution
# -----------------------------
def plot_numeric_distribution(df, column, bins=50):
    data = df[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(data, bins=bins, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Distribution of {_pretty(column)}")
    axes[0].set_xlabel(_pretty(column))
    axes[0].set_ylabel("Frequency")

    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f"{_pretty(column)} Box Plot")
    axes[1].set_ylabel(_pretty(column))

    plt.tight_layout()
    plt.show()

    return data.describe()


# -----------------------------
# Categorical distribution
# -----------------------------
def plot_categorical_distribution(df, cat_cols, top_n=20):
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cat_cols):
        vc = df[col].value_counts().head(top_n)
        ax.bar(vc.index, vc.values, edgecolor="black")
        ax.set_title(f"{_pretty(col)} Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Numeric by class
# -----------------------------
def plot_numeric_by_class(df, column, class_col="class"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df.boxplot(column=column, by=class_col, ax=axes[0])
    axes[0].set_title(f"{_pretty(column)} by Class")
    axes[0].set_xlabel("Class (0=Non-Fraud, 1=Fraud)")
    axes[0].set_ylabel(_pretty(column))
    plt.suptitle("")

    data = [
        df.loc[df[class_col] == 0, column].dropna(),
        df.loc[df[class_col] == 1, column].dropna(),
    ]

    axes[1].violinplot(data, showmeans=True)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Non-Fraud", "Fraud"])
    axes[1].set_title(f"{_pretty(column)} Distribution by Class")
    axes[1].set_ylabel(_pretty(column))

    plt.tight_layout()
    plt.show()


# -----------------------------
# Fraud rate by category
# -----------------------------


def plot_country_fraud_overview(
    country_stats,
    country_stats_filtered,
    overall_rate,
    top_n=10,
    min_transactions=None,
):
    """
    Visualize fraud by country using:
    1) Top countries by fraud count
    2) Top countries by fraud rate (filtered by min transactions)
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # -----------------------------
    # Top countries by fraud count
    # -----------------------------
    top_by_count = country_stats.head(top_n)

    axes[0].barh(
        top_by_count["country"],
        top_by_count["fraud_count"],
        edgecolor="black",
    )
    axes[0].set_xlabel("Fraud Count")
    axes[0].set_ylabel("Country")
    axes[0].set_title(f"Top {top_n} Countries by Fraud Count")
    axes[0].invert_yaxis()

    # -----------------------------
    # Top countries by fraud rate
    # -----------------------------
    top_by_rate = (
        country_stats_filtered
        .sort_values("fraud_rate", ascending=False)
        .head(top_n)
    )

    axes[1].barh(
        top_by_rate["country"],
        top_by_rate["fraud_rate"] * 100,
        edgecolor="black",
    )
    axes[1].set_xlabel("Fraud Rate (%)")
    axes[1].set_ylabel("Country")

    subtitle = f"Top {top_n} Countries by Fraud Rate"
    if min_transactions:
        subtitle += f" (min {min_transactions} txns)"

    axes[1].set_title(subtitle)
    axes[1].invert_yaxis()

    # Overall fraud rate reference
    axes[1].axvline(
        overall_rate,
        color="black",
        linestyle="--",
        label=f"Overall: {overall_rate:.2f}%",
    )
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Country transactions (stacked)
# -----------------------------


def plot_country_transactions(country_stats, top_n=20):
    top = country_stats.head(top_n).copy()

    non_fraud = top["total_transactions"] - top["fraud_count"]
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, non_fraud, label="Non-Fraud")
    ax.bar(x, top["fraud_count"], bottom=non_fraud, label="Fraud")

    ax.set_xticks(x)
    ax.set_xticklabels(top["country"], rotation=45, ha="right")
    ax.set_ylabel("Transaction Count")
    ax.set_title(f"Transaction Distribution by Country (Top {top_n})")
    ax.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot fraud over time
# -----------------------------

def plot_fraud_over_time(df, time_col='purchase_time', freq='D'):
    """
    Plot fraud counts and fraud rate over time using existing time_col.

    Parameters
    ----------
    df : pd.DataFrame
        Must include time_col and 'class'.
    time_col : str
        Column to use for time-series aggregation (e.g., 'purchase_time').
    freq : str
        Resampling frequency: 'D' = daily, 'W' = weekly, 'M' = monthly.
    """
    ts = df.set_index(time_col).resample(freq)
    total_txn = ts['class'].count()
    fraud_txn = ts['class'].sum()
    fraud_rate = (fraud_txn / total_txn) * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(total_txn.index, total_txn.values,
            color='skyblue', label='Total Transactions')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total Transactions', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.plot(fraud_rate.index, fraud_rate.values,
             color='red', marker='o', label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle(
        f'Transactions and Fraud Rate over Time ({freq} frequency)', fontsize=16)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot fraud by hour of day / day of week
# -----------------------------
def plot_fraud_by_hour_day(df):
    """
    Plot fraud counts and fraud rate using existing hour_of_day and day_of_week columns.
    """
    # Hour of day
    hourly = df.groupby('hour_of_day')['class'].agg(['count', 'sum'])
    hourly['fraud_rate'] = (hourly['sum'] / hourly['count'])*100

    # Day of week
    daily = df.groupby('day_of_week')['class'].agg(['count', 'sum'])
    daily['fraud_rate'] = (daily['sum'] / daily['count'])*100

    # Optional: ensure days are ordered correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = daily.reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Hourly
    axes[0].bar(hourly.index, hourly['count'], color='skyblue',
                alpha=0.7, label='Total Transactions')
    axes[0].plot(hourly.index, hourly['fraud_rate'],
                 color='red', marker='o', label='Fraud Rate (%)')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Count / Fraud Rate')
    axes[0].set_title('Transactions and Fraud Rate by Hour of Day')
    axes[0].legend()

    # Day of week
    axes[1].bar(daily.index, daily['count'], color='skyblue',
                alpha=0.7, label='Total Transactions')
    axes[1].plot(daily.index, daily['fraud_rate'], color='red',
                 marker='o', label='Fraud Rate (%)')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Count / Fraud Rate')
    axes[1].set_title('Transactions and Fraud Rate by Day of Week')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
