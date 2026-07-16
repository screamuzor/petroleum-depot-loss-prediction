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