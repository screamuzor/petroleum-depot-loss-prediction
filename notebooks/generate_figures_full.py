import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────
BASE = r'C:\Users\6USER\Desktop\D2 REPORT 2025'
OUT  = BASE

ago_raw = pd.read_csv(os.path.join(BASE, 'ago_m2.csv'),    parse_dates=['Date'])
pms_raw = pd.read_csv(os.path.join(BASE, 'pms_m2.csv'),    parse_dates=['Date'])
tank_df = pd.read_csv(os.path.join(BASE, 'tank_clean.csv'), parse_dates=['Date'])

print(f"AGO: {len(ago_raw)} | PMS: {len(pms_raw)} | Tank: {len(tank_df)}")

# ── Colours ───────────────────────────────────────────────────────
AGO_C = '#1A5F9E'
PMS_C = '#C94E2D'

# ── Feature sets ──────────────────────────────────────────────────
M2_FEATURES = ['Opening_Stock', 'Total_Receipts', 'Total_Deliveries',
                'Receipt_Delivery_Ratio', 'Lag1_Loss', 'Lag2_Loss',
                'Rolling3_Loss', 'DayOfWeek', 'Month']

TANK_FEATURES = ['Opening_Dip_mm', 'Water_Dip_mm', 'Temperature',
                 'Opening_Vol', 'Receipt', 'Dispatch', 'Has_Water',
                 'Dip_Change', 'Lag1_LG', 'Lag2_LG', 'Rolling3_LG',
                 'Has_Receipt', 'Has_Dispatch', 'DayOfWeek', 'Month_Num']

TANK_LABELS = ['Opening Dip (mm)', 'Water Dip (mm)', 'Temperature (C)',
               'Opening Volume', 'Receipt', 'Dispatch', 'Has Water',
               'Dip Change', 'Lag-1 L/G', 'Lag-2 L/G', 'Rolling 3-Day L/G',
               'Has Receipt', 'Has Dispatch', 'Day of Week', 'Month']

# ── Engineer M2 features ──────────────────────────────────────────
def engineer_m2(df):
    df = df.copy()
    df['Total_Receipts']         = df['Total_Receipts'].fillna(0)
    df['Total_Deliveries']       = df['Total_Deliveries'].fillna(0)
    df['Receipt_Delivery_Ratio'] = np.where(df['Total_Deliveries'] > 0,
                                    df['Total_Receipts'] / (df['Total_Deliveries'] + 1), 0)
    df['Lag1_Loss']    = df['Loss_Gain'].shift(1).fillna(0)
    df['Lag2_Loss']    = df['Loss_Gain'].shift(2).fillna(0)
    df['Rolling3_Loss']= df['Loss_Gain'].shift(1).rolling(3, min_periods=1).mean().fillna(0)
    df['DayOfWeek']    = df['Date'].dt.dayofweek
    df['Month']        = df['Date'].dt.month
    df['Loss_Flag']    = (df['Loss_Gain'] < 0).astype(int)
    return df

# ── Engineer tank features ────────────────────────────────────────
def engineer_tank(df):
    df = df.sort_values(['Tank', 'Date']).reset_index(drop=True)
    df['Has_Water']    = (df['Water_Dip_mm'] > 0).astype(int)
    df['Dip_Change']   = df.groupby('Tank')['Opening_Dip_mm'].diff().fillna(0)
    df['Lag1_LG']      = df.groupby('Tank')['Loss_Gain'].shift(1).fillna(0)
    df['Lag2_LG']      = df.groupby('Tank')['Loss_Gain'].shift(2).fillna(0)
    df['Rolling3_LG']  = (df.groupby('Tank')['Loss_Gain'].shift(1)
                           .transform(lambda x: x.rolling(3, min_periods=1).mean()).fillna(0))
    df['Has_Receipt']  = (df['Receipt'] > 0).astype(int)
    df['Has_Dispatch'] = (df['Dispatch'] > 0).astype(int)
    df['DayOfWeek']    = df['Date'].dt.dayofweek
    df['Month_Num']    = df['Date'].dt.month
    df['Loss_Flag']    = (df['Loss_Gain'] < 0).astype(int)
    return df

ago = engineer_m2(ago_raw)
pms = engineer_m2(pms_raw)
tank_df = engineer_tank(tank_df)

# ── Fit M2 models ─────────────────────────────────────────────────
def fit_m2(df, features):
    n = len(df); split = int(n * 0.8)
    X = df[features]; y_reg = df['Loss_Gain']; y_cls = df['Loss_Flag']
    X_tr, X_te   = X.iloc[:split], X.iloc[split:]
    yr_tr, yr_te = y_reg.iloc[:split], y_reg.iloc[split:]
    yc_tr, yc_te = y_cls.iloc[:split], y_cls.iloc[split:]
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    lr  = LinearRegression().fit(X_tr, yr_tr)
    dtr = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_tr, yr_tr)
    log = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X_tr_sc, yc_tr)
    dtc = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42).fit(X_tr, yc_tr)
    return dict(
        yr_te=yr_te, yc_te=yc_te,
        lr_pred=lr.predict(X_te),
        dtr_pred=dtr.predict(X_te),
        feat_imp=pd.Series(dtr.feature_importances_, index=features).sort_values(ascending=False),
        log_pred=log.predict(X_te_sc), log_prob=log.predict_proba(X_te_sc)[:,1],
        log_auc=roc_auc_score(yc_te, log.predict_proba(X_te_sc)[:,1]),
        cm_log=confusion_matrix(yc_te, log.predict(X_te_sc)),
        dtc_pred=dtc.predict(X_te), dtc_prob=dtc.predict_proba(X_te)[:,1],
        dtc_auc=roc_auc_score(yc_te, dtc.predict_proba(X_te)[:,1]),
        cm_dtc=confusion_matrix(yc_te, dtc.predict(X_te)),
        split=split, dates=df['Date']
    )

# ── Fit tank models ───────────────────────────────────────────────
def fit_tank(product):
    sub = tank_df[tank_df['Product'] == product].copy()
    sub = sub.sort_values('Date').reset_index(drop=True)
    sub = sub.dropna(subset=TANK_FEATURES + ['Loss_Gain', 'Loss_Flag'])
    n = len(sub); split = int(n * 0.8)
    X = sub[TANK_FEATURES]; y_reg = sub['Loss_Gain']; y_cls = sub['Loss_Flag']
    X_tr, X_te   = X.iloc[:split], X.iloc[split:]
    yr_tr, yr_te = y_reg.iloc[:split], y_reg.iloc[split:]
    yc_tr, yc_te = y_cls.iloc[:split], y_cls.iloc[split:]
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    lr  = LinearRegression().fit(X_tr, yr_tr)
    dtr = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_tr, yr_tr)
    log = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X_tr_sc, yc_tr)
    dtc = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42).fit(X_tr, yc_tr)
    return dict(
        sub=sub, yr_te=yr_te, yc_te=yc_te,
        lr_pred=lr.predict(X_te),
        dtr_pred=dtr.predict(X_te),
        feat_imp=pd.Series(dtr.feature_importances_, index=TANK_LABELS).sort_values(ascending=False),
        log_pred=log.predict(X_te_sc), log_prob=log.predict_proba(X_te_sc)[:,1],
        log_auc=roc_auc_score(yc_te, log.predict_proba(X_te_sc)[:,1]),
        cm_log=confusion_matrix(yc_te, log.predict(X_te_sc)),
        dtc_pred=dtc.predict(X_te), dtc_prob=dtc.predict_proba(X_te)[:,1],
        dtc_auc=roc_auc_score(yc_te, dtc.predict_proba(X_te)[:,1]),
        cm_dtc=confusion_matrix(yc_te, dtc.predict(X_te))
    )

print("Fitting models...")
m2  = {'AGO': fit_m2(ago, M2_FEATURES), 'PMS': fit_m2(pms, M2_FEATURES)}
tkm = {'AGO': fit_tank('AGO'),           'PMS': fit_tank('PMS')}
print("Models fitted. Generating figures...")

# ================================================================
# FIGURE 5.1 — Depot-level Regression
# ================================================================
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.patch.set_facecolor('white')
fig.suptitle('Figure 5.1 — Depot-Level Regression Performance (80/20 Split)\n'
             'APD Depot 2025 | Linear Regression & Decision Tree Regressor',
             fontsize=13, fontweight='bold', y=1.01)

configs = [
    ('AGO', AGO_C, 'Linear Regression',      m2['AGO']['lr_pred']),
    ('AGO', AGO_C, 'Decision Tree Regressor', m2['AGO']['dtr_pred']),
    ('PMS', PMS_C, 'Linear Regression',       m2['PMS']['lr_pred']),
    ('PMS', PMS_C, 'Decision Tree Regressor', m2['PMS']['dtr_pred']),
]

for col, (prod, col_c, mname, pred) in enumerate(configs):
    yr    = m2[prod]['yr_te']
    r2    = r2_score(yr, pred)
    mae   = mean_absolute_error(yr, pred)
    rmse  = np.sqrt(mean_squared_error(yr, pred))
    resid = yr.values - pred

    ax = axes[0, col]
    ax.scatter(yr, pred, alpha=0.45, s=20, color=col_c, edgecolors='none')
    lims = [min(yr.min(), pred.min())-5000, max(yr.max(), pred.max())+5000]
    ax.plot(lims, lims, 'r--', lw=1.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual L/G (litres)', fontsize=9)
    ax.set_ylabel('Predicted L/G (litres)', fontsize=9)
    ax.set_title(f'{prod} — {mname}', fontsize=10, fontweight='bold', color=col_c)
    ax.text(0.04, 0.96, f'R2 = {r2:.4f}\nMAE = {mae:,.0f} L\nRMSE = {rmse:,.0f} L',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', alpha=0.85))
    ax.grid(True, alpha=0.25); ax.set_facecolor('#FAFAFA')

    ax2 = axes[1, col]
    ax2.axhline(0, color='red', lw=1.2, linestyle='--')
    ax2.scatter(pred, resid, alpha=0.45, s=18, color=col_c, edgecolors='none')
    ax2.set_xlabel('Predicted (litres)', fontsize=9)
    ax2.set_ylabel('Residual (litres)', fontsize=9)
    ax2.set_title(f'Residuals — {prod} {mname}', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.25); ax2.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Fig5_1_Regression.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.1 saved")

# ================================================================
# FIGURE 5.2 — Depot-level Classification
# ================================================================
fig2 = plt.figure(figsize=(24, 10))
fig2.patch.set_facecolor('white')
fig2.suptitle('Figure 5.2 — Depot-Level Classification Performance (80/20 Split)\n'
              'APD Depot 2025 | Logistic Regression & Decision Tree Classifier',
              fontsize=13, fontweight='bold', y=1.01)

gs = gridspec.GridSpec(2, 6, figure=fig2, hspace=0.5, wspace=0.45)

cls_conf = [
    ('AGO', AGO_C, 'Logistic Regression',     m2['AGO']['log_pred'], m2['AGO']['log_prob'], m2['AGO']['cm_log'], m2['AGO']['log_auc']),
    ('AGO', AGO_C, 'Decision Tree Classifier', m2['AGO']['dtc_pred'], m2['AGO']['dtc_prob'], m2['AGO']['cm_dtc'], m2['AGO']['dtc_auc']),
    ('PMS', PMS_C, 'Logistic Regression',      m2['PMS']['log_pred'], m2['PMS']['log_prob'], m2['PMS']['cm_log'], m2['PMS']['log_auc']),
    ('PMS', PMS_C, 'Decision Tree Classifier', m2['PMS']['dtc_pred'], m2['PMS']['dtc_prob'], m2['PMS']['cm_dtc'], m2['PMS']['dtc_auc']),
]

positions = [(0,0),(0,2),(0,4),(1,0)]
for idx, (prod, col_c, mname, pred, prob, cm, auc) in enumerate(cls_conf):
    row, c = positions[idx]
    ax = fig2.add_subplot(gs[row, c:c+2])
    cmap = plt.cm.Blues if prod == 'AGO' else plt.cm.Oranges
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(f'{prod} — {mname}\nAUC = {auc:.4f}', fontsize=10, fontweight='bold', color=col_c)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Gain/\nNeutral','Loss'], fontsize=9)
    ax.set_yticklabels(['Gain/\nNeutral','Loss'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')

for ri, (prod, col_c) in enumerate([('AGO', AGO_C), ('PMS', PMS_C)]):
    m  = m2[prod]
    ax = fig2.add_subplot(gs[1, ri*2+2:ri*2+4])
    ax.set_facecolor('#FAFAFA')
    for mname, prob, ls in [('Logistic Regression', m['log_prob'], '-'),
                              ('Decision Tree Classifier', m['dtc_prob'], '--')]:
        fpr, tpr, _ = roc_curve(m['yc_te'], prob)
        auc_v = roc_auc_score(m['yc_te'], prob)
        ax.plot(fpr, tpr, lw=2.2, linestyle=ls, color=col_c,
                label=f'{mname} (AUC={auc_v:.3f})', alpha=0.85)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title(f'{prod} — ROC Curve', fontsize=10, fontweight='bold', color=col_c)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

fig2.savefig(os.path.join(OUT, 'Fig5_2_Classification.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.2 saved")

# ================================================================
# FIGURE 5.3 — Feature Importance + Pareto
# ================================================================
fig3, axes = plt.subplots(2, 2, figsize=(22, 12))
fig3.patch.set_facecolor('white')
fig3.suptitle('Figure 5.3 — Depot-Level Feature Importance & Pareto Analysis\n'
              'APD Depot 2025 | Decision Tree Regressor | 80/20 Rule',
              fontsize=13, fontweight='bold', y=1.01)

for row, (prod, col_c, df_raw) in enumerate([('AGO', AGO_C, ago), ('PMS', PMS_C, pms)]):
    fi = m2[prod]['feat_imp'].sort_values(ascending=True)
    ax = axes[row, 0]
    bars = ax.barh(fi.index, fi.values, color=col_c, alpha=0.82, edgecolor='white', height=0.65)
    for bar, val in zip(bars, fi.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=8.5, color='#444')
    ax.set_title(f'{prod} — Feature Importance', fontsize=11, fontweight='bold', color=col_c)
    ax.set_xlabel('Importance Score', fontsize=9)
    ax.set_xlim(0, fi.max()*1.22)
    ax.grid(axis='x', alpha=0.25); ax.set_facecolor('#FAFAFA')
    ax.spines[['top','right']].set_visible(False)

    losses     = df_raw[df_raw['Loss_Gain'] < 0].copy()
    losses['Abs_Loss'] = losses['Loss_Gain'].abs()
    total_loss = losses['Abs_Loss'].sum()
    ls         = losses.sort_values('Abs_Loss', ascending=False).reset_index(drop=True)
    ls['CumPct'] = ls['Abs_Loss'].cumsum() / total_loss * 100
    n_80 = (ls['CumPct'] <= 80).sum()

    ax2 = axes[row, 1]
    ax3 = ax2.twinx()
    ax2.bar(range(len(ls)), ls['Abs_Loss']/1000, color=col_c, alpha=0.7, edgecolor='none', width=0.9)
    ax3.plot(range(len(ls)), ls['CumPct'], color='#E67E22', lw=2.2, marker='o', markersize=3)
    ax3.axhline(80, color='red', lw=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(n_80-0.5, color='red', lw=1.5, linestyle='--', alpha=0.8)
    ax2.set_xlabel('Loss Events (ranked by magnitude)', fontsize=9)
    ax2.set_ylabel('Loss Volume (000 litres)', fontsize=9, color=col_c)
    ax3.set_ylabel('Cumulative % of Total Loss', fontsize=9, color='#E67E22')
    ax3.set_ylim(0, 110); ax3.tick_params(axis='y', colors='#E67E22')
    ax2.set_title(f'{prod} — Pareto: Top {n_80} events ({n_80/len(ls)*100:.0f}%) = 80% of loss',
                  fontsize=10, fontweight='bold', color=col_c)
    ax2.set_facecolor('#FAFAFA'); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(OUT, 'Fig5_3_FeatureImportance_Pareto.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.3 saved")

# ================================================================
# FIGURE 5.4 — Time Series
# ================================================================
fig4, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=False)
fig4.patch.set_facecolor('white')
fig4.suptitle('Figure 5.4 — Daily Loss/Gain Time Series: AGO & PMS (January to December 2025)\n'
              'APD Depot | Training Period vs Test Period (shaded)',
              fontsize=13, fontweight='bold', y=1.01)

for ax, df_raw, prod, col_c in [(axes[0], ago, 'AGO', AGO_C), (axes[1], pms, 'PMS', PMS_C)]:
    split_date = df_raw['Date'].iloc[int(len(df_raw)*0.8)]
    gains  = df_raw[df_raw['Loss_Gain'] >= 0]
    losses = df_raw[df_raw['Loss_Gain'] < 0]
    ax.bar(gains['Date'],  gains['Loss_Gain']/1000,  color='#27AE60', alpha=0.75, width=0.9, label='Gain (K litres)')
    ax.bar(losses['Date'], losses['Loss_Gain']/1000, color=col_c,     alpha=0.80, width=0.9, label='Loss (K litres)')
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(split_date, color='navy', lw=2, linestyle='--', alpha=0.8,
               label=f'Train/Test Split ({split_date.strftime("%d %b %Y")})')
    ax.axvspan(split_date, df_raw['Date'].iloc[-1], alpha=0.06, color='navy')
    ax.set_ylabel('Loss / Gain (000 litres)', fontsize=10)
    ax.set_title(f'{prod} — Daily Stock Gain/Loss 2025', fontsize=11, fontweight='bold', color=col_c)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25); ax.set_facecolor('#FAFAFA')
    ax.spines[['top','right']].set_visible(False)
    min_row = df_raw.loc[df_raw['Loss_Gain'].idxmin()]
    ax.annotate(f'Largest loss\n{min_row["Loss_Gain"]/1000:,.1f}K L\n{min_row["Date"].strftime("%d %b")}',
                xy=(min_row['Date'], min_row['Loss_Gain']/1000),
                xytext=(min_row['Date'], min_row['Loss_Gain']/1000 - 8),
                fontsize=8, color='red', ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

plt.tight_layout()
fig4.savefig(os.path.join(OUT, 'Fig5_4_TimeSeries.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.4 saved")

# ================================================================
# FIGURE 5.5 — Tank-level Regression
# ================================================================
fig5, axes = plt.subplots(2, 4, figsize=(24, 10))
fig5.patch.set_facecolor('white')
fig5.suptitle('Figure 5.5 — Tank-Level Regression Performance (80/20 Split)\n'
              'APD Depot 2025 | Features include Opening Dip (mm), Water Dip, Temperature',
              fontsize=13, fontweight='bold', y=1.01)

tk_configs = [
    ('AGO', AGO_C, 'Linear Regression',      tkm['AGO']['lr_pred']),
    ('AGO', AGO_C, 'Decision Tree Regressor', tkm['AGO']['dtr_pred']),
    ('PMS', PMS_C, 'Linear Regression',       tkm['PMS']['lr_pred']),
    ('PMS', PMS_C, 'Decision Tree Regressor', tkm['PMS']['dtr_pred']),
]

for col, (prod, col_c, mname, pred) in enumerate(tk_configs):
    yr    = tkm[prod]['yr_te']
    r2    = r2_score(yr, pred)
    mae   = mean_absolute_error(yr, pred)
    rmse  = np.sqrt(mean_squared_error(yr, pred))
    resid = yr.values - pred

    ax = axes[0, col]
    ax.scatter(yr, pred, alpha=0.45, s=20, color=col_c, edgecolors='none')
    lims = [min(yr.min(), pred.min())-5000, max(yr.max(), pred.max())+5000]
    ax.plot(lims, lims, 'r--', lw=1.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual L/G (litres)', fontsize=9)
    ax.set_ylabel('Predicted L/G (litres)', fontsize=9)
    ax.set_title(f'{prod} — {mname}', fontsize=10, fontweight='bold', color=col_c)
    ax.text(0.04, 0.96, f'R2 = {r2:.4f}\nMAE = {mae:,.0f} L\nRMSE = {rmse:,.0f} L',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', alpha=0.85))
    ax.grid(True, alpha=0.25); ax.set_facecolor('#FAFAFA')

    ax2 = axes[1, col]
    ax2.axhline(0, color='red', lw=1.2, linestyle='--')
    ax2.scatter(pred, resid, alpha=0.45, s=18, color=col_c, edgecolors='none')
    ax2.set_xlabel('Predicted (litres)', fontsize=9)
    ax2.set_ylabel('Residual (litres)', fontsize=9)
    ax2.set_title(f'Residuals — {prod} {mname}', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.25); ax2.set_facecolor('#FAFAFA')

plt.tight_layout()
fig5.savefig(os.path.join(OUT, 'Fig5_5_TankLevel_Regression.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.5 saved")

# ================================================================
# FIGURE 5.6 — Tank-level Classification
# ================================================================
fig6 = plt.figure(figsize=(24, 10))
fig6.patch.set_facecolor('white')
fig6.suptitle('Figure 5.6 — Tank-Level Classification Performance (80/20 Split)\n'
              'APD Depot 2025 | Logistic Regression & Decision Tree Classifier',
              fontsize=13, fontweight='bold', y=1.01)

gs2 = gridspec.GridSpec(2, 6, figure=fig6, hspace=0.5, wspace=0.45)

tk_cls = [
    ('AGO', AGO_C, 'Logistic Regression',     tkm['AGO']['log_pred'], tkm['AGO']['log_prob'], tkm['AGO']['cm_log'], tkm['AGO']['log_auc']),
    ('AGO', AGO_C, 'Decision Tree Classifier', tkm['AGO']['dtc_pred'], tkm['AGO']['dtc_prob'], tkm['AGO']['cm_dtc'], tkm['AGO']['dtc_auc']),
    ('PMS', PMS_C, 'Logistic Regression',      tkm['PMS']['log_pred'], tkm['PMS']['log_prob'], tkm['PMS']['cm_log'], tkm['PMS']['log_auc']),
    ('PMS', PMS_C, 'Decision Tree Classifier', tkm['PMS']['dtc_pred'], tkm['PMS']['dtc_prob'], tkm['PMS']['cm_dtc'], tkm['PMS']['dtc_auc']),
]

for idx, (prod, col_c, mname, pred, prob, cm, auc) in enumerate(tk_cls):
    row, c = positions[idx]
    ax = fig6.add_subplot(gs2[row, c:c+2])
    cmap = plt.cm.Blues if prod == 'AGO' else plt.cm.Oranges
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(f'{prod} — {mname}\nAUC = {auc:.4f}', fontsize=10, fontweight='bold', color=col_c)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Gain/\nNeutral','Loss'], fontsize=9)
    ax.set_yticklabels(['Gain/\nNeutral','Loss'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')

for ri, (prod, col_c) in enumerate([('AGO', AGO_C), ('PMS', PMS_C)]):
    m  = tkm[prod]
    ax = fig6.add_subplot(gs2[1, ri*2+2:ri*2+4])
    ax.set_facecolor('#FAFAFA')
    for mname, prob, ls in [('Logistic Regression', m['log_prob'], '-'),
                              ('Decision Tree Classifier', m['dtc_prob'], '--')]:
        fpr, tpr, _ = roc_curve(m['yc_te'], prob)
        auc_v = roc_auc_score(m['yc_te'], prob)
        ax.plot(fpr, tpr, lw=2.2, linestyle=ls, color=col_c,
                label=f'{mname} (AUC={auc_v:.3f})', alpha=0.85)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title(f'{prod} — ROC Curve (Tank-Level)', fontsize=10, fontweight='bold', color=col_c)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

fig6.savefig(os.path.join(OUT, 'Fig5_6_TankLevel_Classification.png'), dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.6 saved")

# ================================================================
# FIGURE 5.7 — Tank Feature Importance + Dip Threshold
# ================================================================
fig7, axes = plt.subplots(2, 2, figsize=(22, 13))
fig7.patch.set_facecolor('white')
fig7.suptitle('Figure 5.7 — Tank-Level Feature Importance & Dip Threshold Loss Probability\n'
              'APD Depot 2025 | Operational Recommendation: Safe vs High-Risk Dip Levels',
              fontsize=13, fontweight='bold', y=1.01)

bins_dip  = [0, 1000, 2000, 3000, 5000, 7000, 9000, 15000]
labels_dip = ['0-1K','1-2K','2-3K','3-5K','5-7K','7-9K','9K+']

for row, (prod, col_c) in enumerate([('AGO', AGO_C), ('PMS', PMS_C)]):
    fi = tkm[prod]['feat_imp'].sort_values(ascending=True)
    ax = axes[row, 0]
    bars = ax.barh(fi.index, fi.values, color=col_c, alpha=0.82, edgecolor='white', height=0.65)
    for bar, val in zip(bars, fi.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=8.5, color='#444')
    ax.set_title(f'{prod} — Feature Importance (Decision Tree Regressor)',
                 fontsize=11, fontweight='bold', color=col_c)
    ax.set_xlabel('Importance Score', fontsize=9)
    ax.set_xlim(0, fi.max()*1.22)
    ax.grid(axis='x', alpha=0.25); ax.set_facecolor('#FAFAFA')
    ax.spines[['top','right']].set_visible(False)

    ax2 = axes[row, 1]
    sub = tank_df[(tank_df['Product'] == prod) & (tank_df['Opening_Dip_mm'] > 0)].copy()
    sub['Band'] = pd.cut(sub['Opening_Dip_mm'], bins=bins_dip, labels=labels_dip)
    tbl = sub.groupby('Band', observed=True).agg(
        Count=('Loss_Flag','count'),
        Loss_Prob=('Loss_Flag','mean')
    ).reset_index()
    tbl['Loss_Prob_Pct'] = tbl['Loss_Prob'] * 100
    bar_colors = ['#E74C3C' if p > 20 else '#E67E22' if p > 12 else '#27AE60'
                  for p in tbl['Loss_Prob_Pct']]
    bars2 = ax2.bar(tbl['Band'], tbl['Loss_Prob_Pct'], color=bar_colors,
                    alpha=0.85, edgecolor='white', width=0.7)
    for bar, (_, r) in zip(bars2, tbl.iterrows()):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f'{r["Loss_Prob_Pct"]:.1f}%\n(n={r["Count"]})',
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.axhline(15, color='orange', lw=1.5, linestyle='--', alpha=0.8, label='15% threshold')
    ax2.axhline(20, color='red',    lw=1.5, linestyle='--', alpha=0.8, label='20% high risk')
    ax2.set_xlabel('Opening Dip Range (mm)', fontsize=10)
    ax2.set_ylabel('Loss Probability (%)', fontsize=10)
    ax2.set_title(f'{prod} — Loss Probability by Opening Dip Level\n'
                  'Red = High Risk (>20%) | Orange = Moderate | Green = Low',
                  fontsize=11, fontweight='bold', color=col_c)
    ax2.legend(fontsize=8.5, loc='upper right')
    ax2.set_ylim(0, max(tbl['Loss_Prob_Pct'].max()+8, 30))
    ax2.grid(axis='y', alpha=0.25); ax2.set_facecolor('#FAFAFA')
    ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
fig7.savefig(os.path.join(OUT, 'Fig5_7_DipThreshold_FeatureImportance.png'),
             dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 5.7 saved")

print("\nAll 7 figures generated and saved to:")
print(f"  {OUT}")
