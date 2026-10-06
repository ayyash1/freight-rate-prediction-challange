"""
Generates publication-quality figures for the Freight Rate ML Assessment Report.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Setup style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dir = Path("report/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Feature Importance Chart
fi_path = Path("artifacts/feature_importance.csv")
if fi_path.exists():
    fi_df = pd.read_csv(fi_path).head(12)
    plt.figure(figsize=(9, 5), dpi=200)
    colors = sns.color_palette("mako", len(fi_df))
    plt.barh(fi_df['feature'][::-1], fi_df['importance'][::-1], color=colors)
    plt.title("Top 12 Most Influential Features (LightGBM GBDT)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Feature Importance (Split Gain / Count)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", bbox_inches='tight')
    plt.close()
    print("Created feature_importance.png")

# 2. Freight Rate vs Distance by Equipment
train_df = pd.read_csv("data/train_test.csv")
train_df['weight_clean'] = train_df['weight'].abs().fillna(31500)
train_df['rpm'] = train_df['posted_rate'] / train_df['distance']

plt.figure(figsize=(9, 5), dpi=200)
for eq, col in zip(['Dry Van', 'Flatbed', 'Reefer'], ['#1f77b4', '#2ca02c', '#d62728']):
    sub = train_df[train_df['equipment'] == eq]
    plt.scatter(sub['distance'].iloc[:800], sub['posted_rate'].iloc[:800], alpha=0.35, s=16, label=eq, color=col)
plt.title("Posted Load Rate vs Distance by Equipment Type", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Haul Distance (miles)", fontsize=11)
plt.ylabel("Posted Rate ($)", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(fig_dir / "rate_vs_distance.png", bbox_inches='tight')
plt.close()
print("Created rate_vs_distance.png")

# 3. Rate Per Mile vs Weight
plt.figure(figsize=(9, 4.8), dpi=200)
sns.boxplot(x=pd.qcut(train_df['weight_clean'], 5, labels=['<25k', '25k-29k', '29k-33k', '33k-38k', '>38k']),
            y=train_df['rpm'], palette="crest", showfliers=False)
plt.title("Rate Per Mile ($/mi) Distribution by Cargo Weight Tier", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Weight Bracket (lbs)", fontsize=11)
plt.ylabel("Rate Per Mile ($/mi)", fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / "rpm_by_weight_tier.png", bbox_inches='tight')
plt.close()
print("Created rpm_by_weight_tier.png")

# 4. Monthly & Day-of-Week Seasonality
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['day_name'] = train_df['date'].dt.day_name()
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_avg = train_df.groupby('day_name')['rpm'].mean().reindex(dow_order)

plt.figure(figsize=(8, 4.5), dpi=200)
plt.plot(dow_order, dow_avg.values, marker='o', linewidth=2.5, markersize=7, color='#064A56')
plt.title("Average Rate Per Mile ($/mi) by Day of Week", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Day of Week", fontsize=11)
plt.ylabel("Average RPM ($/mi)", fontsize=11)
plt.ylim(dow_avg.min() * 0.98, dow_avg.max() * 1.02)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(fig_dir / "day_of_week_seasonality.png", bbox_inches='tight')
plt.close()
print("Created day_of_week_seasonality.png")

print("All report figures generated successfully.")
