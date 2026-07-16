# SQL Scripts — APD Petroleum Depot Loss Prediction

This folder contains the PostgreSQL scripts used to create, load, and query the operational datasets for this study.

## Database

- **Database name:** oil_gas_db
- **Schema:** public
- **Engine:** PostgreSQL 18
- **Tool:** pgAdmin 4

---

## Tables

### 1. depot_operations
Depot-level daily stock summary extracted from the M2 monthly report workbook.

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| date | DATE | Operational date |
| product | VARCHAR(3) | AGO or PMS |
| opening_stock | NUMERIC | Opening stock in litres |
| vessel_receipt_volume | NUMERIC | Vessel receipt in litres |
| external_pipeline_receipt | NUMERIC | Pipeline receipt in litres |
| total_receipts | NUMERIC | Total receipts in litres |
| total_deliveries | NUMERIC | Total dispatches in litres |
| book_balance | NUMERIC | Computed book balance in litres |
| closing_stock | NUMERIC | Physical closing stock in litres |
| loss_gain | NUMERIC | Daily Loss/Gain in litres |
| loss_gain_pct | NUMERIC | Loss/Gain as percentage of opening stock |

**Rows:** 730 (365 AGO + 365 PMS, January to December 2025)

---

### 2. tank_operations
Individual tank daily dip records extracted from D2 daily operational reports.

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| date | DATE | Operational date |
| tank_id | VARCHAR(10) | Tank identifier (T-102, T-121 etc.) |
| product | VARCHAR(3) | AGO or PMS |
| opening_dip_mm | NUMERIC | Opening gross dip in millimetres |
| water_dip_mm | NUMERIC | Water layer at tank bottom in millimetres |
| temperature_c | NUMERIC | Product temperature in degrees Celsius |
| opening_volume | NUMERIC | Opening product volume in litres |
| receipt | NUMERIC | Litres received into tank |
| dispatch | NUMERIC | Litres dispatched from tank |
| closing_dip_mm | NUMERIC | Closing gross dip in millimetres |
| loss_gain | NUMERIC | Tank-level Loss/Gain in litres |
| loss_gain_pct | NUMERIC | Loss/Gain as percentage of opening volume |
| month_name | VARCHAR(10) | Month label |

**Rows:** 4,264 (February to December 2025, 13 tanks)

**Tanks covered:**
- AGO: T-102, T-121, T-122, T-141, T-142, T-5802, T-5803
- PMS: T-123, T-124, T-143, T-144, T-5801, T-5804

---

## Scripts

### tank_operations.sql
Complete script for the tank_operations table including:
- CREATE TABLE statement
- COPY/Import command to load from CSV
- Verification queries (row counts, date ranges, null checks)
- Validation queries (Loss/Gain summary, dip range distribution)
- Analysis queries used in the thesis (Pareto analysis, water dip analysis)
- Cross-reference query joining tank_operations with depot_operations

---

## How to Run

1. Open pgAdmin 4 and connect to oil_gas_db
2. Open the Query Tool
3. Run the CREATE TABLE section first
4. Use pgAdmin Import/Export to load tank_operations.csv
5. Run the verification queries to confirm 4,264 rows loaded
6. Run the analysis queries to reproduce thesis findings

---

## Relationship Between Tables

```
depot_operations (M2 Summary)
    date + product → aggregated daily depot total

tank_operations (D2 Tank Level)
    date + tank_id + product → individual tank readings

JOIN on: date + product
```

The depot_operations table represents the rolled-up M2 summary.
The tank_operations table represents the granular D2 tank readings.
Individual tank gains and losses cancel each other out before reaching the M2 figure,
which is why classification models trained on tank_operations outperform those
trained on depot_operations (AUC 0.8589 vs 0.6842 for AGO).

---

## Author

Uzoma Nnaemeka Eze
MSc Data Science | University of Europe for Applied Sciences, Potsdam
Supervisor: Prof. Dr. Farhan Khan
2026
