"""
Create reduced dataset with top 15 features based on IV analysis
"""

import pandas as pd
import sys

# Best 15 features from IV analysis
best_15_features = [
    'sales_channel_mod',
    'ratio_ob_max3_msf_rc_mod',
    'dpd_amt_l6m_max_mod',
    'max_age_l6m_mod',
    'dpd_cnt_l6m_mod',
    'ratio_ob_avg3_msf_rc_mod',
    'ratio_dpd_ob_l6m_avg_mod',
    'ratio_ob_avg6_cl_mod',
    'ratio_ob_min3_msf_rc_mod',
    'pofc_mod',
    'max_os_l3m_mod',
    'ratio_ob_cm_bill_lm_mod',
    'ave_os_l3m_mod',
    'ar_php_mod',
    'ratio_ob_min3_cl_mod'
]

print("Loading original data...")
df = pd.read_csv('data/uploads/86356ef4-95f9-4617-9c24-8dd3f3495c93.csv')

print(f"Original data: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Keep only best features + segment + target
keep_cols = best_15_features + ['segment', 'target']
df_reduced = df[keep_cols]

print(f"\nReduced data: {df_reduced.shape}")
print(f"Columns: {len(df_reduced.columns)}")
print(f"Features: {len(best_15_features)}")

# Save
output_file = 'data/uploads/data_reduced_15features.csv'
df_reduced.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")
print(f"✓ File size reduced from {df.memory_usage(deep=True).sum()/1024**2:.0f}MB to {df_reduced.memory_usage(deep=True).sum()/1024**2:.0f}MB")
print(f"✓ Reduction: {(1 - df_reduced.memory_usage(deep=True).sum()/df.memory_usage(deep=True).sum())*100:.0f}%")

print("\nTop 15 features selected:")
for i, feat in enumerate(best_15_features, 1):
    print(f"  {i:2d}. {feat}")

