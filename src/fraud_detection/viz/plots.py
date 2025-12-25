import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, theme="whitegrid"):
        """Initializes the visualizer with a consistent professional theme."""
        sns.set_theme(style=theme)
        self.total_color = "#3498db"   # Trust Blue
        self.fraud_color = "#e74c3c"   # Risk Red
        self.neutral_color = "#bdc3c7"  # Grey
        # Unified palette to handle both string '0'/'1' and int 0/1
        self.palette = {0: self.total_color, 1: self.fraud_color,
                        "0": self.total_color, "1": self.fraud_color}

    def analyze_class_distribution(self, df, target_col='class'):
        """Visualizes class imbalance with percentage annotations."""
        stats = df[target_col].value_counts()
        pcts = df[target_col].value_counts(normalize=True) * 100

        plt.figure(figsize=(9, 6))
        ax = sns.countplot(
            data=df,
            x=target_col,
            hue=target_col,
            palette=self.palette,
            dodge=False
        )

        for p in ax.patches:
            height = p.get_height()
            idx = int(p.get_x() + 0.5)
            txt = f'{int(height):,}\n({pcts.iloc[idx]:.2f}%)'
            ax.annotate(txt, (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', fontweight='bold')

        plt.title('Transaction Class Distribution',
                  loc='left', fontsize=16, pad=20)
        plt.ylabel('Count')
        plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
        plt.ylim(0, stats.max() * 1.2)
        sns.despine()
        plt.show()

    def plot_fraud_distributions(self, df):
        """Plots histograms for Purchase Value and Age segmented by class."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Purchase Value
        sns.histplot(
            data=df,
            x='purchase_value',
            hue='class',
            kde=True,
            element="step",
            palette=self.palette,
            ax=axes[0]
        )
        axes[0].set_title('Distribution of Purchase Value', fontsize=15)

        # Plot 2: Age
        sns.histplot(
            data=df,
            x='age',
            hue='class',
            kde=True,
            element="step",
            palette=self.palette,
            ax=axes[1]
        )
        axes[1].set_title('Distribution of User Age', fontsize=15)

        plt.tight_layout()
        plt.show()

    def plot_purchase_value_boxplot(self, df):
        """Bivariate boxplot with mean markers for Purchase Value analysis."""
        plt.figure(figsize=(10, 7))

        # Boxplot
        sns.boxplot(
            data=df,
            x='class',
            y='purchase_value',
            palette=self.palette,
            linewidth=2
        )

        # Mean markers with pointplot (v0.14+ safe)
        sns.pointplot(
            data=df,
            x='class',
            y='purchase_value',
            estimator=np.mean,
            color='black',
            marker='D',
            linestyle='none'
        )

        plt.title('Purchase Value: Legitimate vs. Fraudulent',
                  fontsize=16, pad=20)
        plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])
        plt.show()

    def plot_top_countries(self, df, top_n=20):
        """Optimized: Uses pre-calculated counts to speed up rendering."""
        counts = df['country'].value_counts().head(top_n).reset_index()
        counts.columns = ['country', 'count']

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            data=counts,
            x='count',
            y='country',
            palette="viridis"
        )

        max_val = counts['count'].max()
        for i, v in enumerate(counts['count']):
            ax.text(v + (max_val * 0.01), i,
                    f'{int(v):,}', va='center', fontsize=10)

        plt.title(f'Top {top_n} Countries by Transaction Volume',
                  loc='left', fontsize=16, pad=20)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    def plot_fraud_rate_by_country(self, df, top_n=10):
        """Optimized: Pre-aggregates mean to avoid heavy Seaborn computation."""
        rates = (
            df.groupby('country')['class']
            .agg(['mean', 'count'])
            .query('count > 10')
            .sort_values(by='mean', ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(10, 7))
        ax = sns.barplot(
            data=rates,
            x='mean',
            y='country',
            palette='Reds_r'
        )

        for i, v in enumerate(rates['mean']):
            ax.text(v + 0.005, i, f'{v:.2%}', va='center',
                    fontweight='bold', color='darkred')

        plt.title(f'Top {top_n} High-Risk Countries (Fraud Rate)',
                  loc='left', fontsize=16, pad=20)
        plt.xlabel('Fraud Probability')
        plt.xlim(0, rates['mean'].max() * 1.2)
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, df):
        """Daily and Hourly time-series analysis for fraud patterns."""
        if not pd.api.types.is_datetime64_any_dtype(df['purchase_time']):
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        fig, axes = plt.subplots(2, 1, figsize=(15, 12))

        # Daily Trend
        daily = df.set_index('purchase_time').resample('D')[
            'class'].agg(['count', 'sum'])
        axes[0].plot(daily.index, daily['count'], label='Total',
                     color=self.total_color, lw=2)
        axes[0].plot(daily.index, daily['sum'], label='Fraud',
                     color=self.fraud_color, lw=2)
        axes[0].set_title('Daily Trends', fontsize=16, loc='left')
        axes[0].legend(frameon=False)

        # Hourly Pattern
        if 'hour_of_day' not in df.columns:
            df['hour_of_day'] = df['purchase_time'].dt.hour

        hourly = df.groupby('hour_of_day')['class'].agg(['count', 'sum'])
        sns.barplot(
            x=hourly.index,
            y=hourly['count'],
            color=self.neutral_color,
            alpha=0.4,
            ax=axes[1],
            label='Total'
        )
        sns.barplot(
            x=hourly.index,
            y=hourly['sum'],
            color=self.fraud_color,
            ax=axes[1],
            label='Fraud'
        )
        axes[1].set_title('Hourly Fraud Seasonality', fontsize=16, loc='left')
        axes[1].legend(frameon=False)

        plt.tight_layout()
        plt.show()
