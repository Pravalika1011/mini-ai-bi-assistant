# Business Metrics & Semantic Glossary

This glossary acts as the semantic layer for the RSG Mini AI Business Intelligence Assistant. It formalizes business definitions, accounting rules, schema terminology, and data governance policies.

---

### Net Revenue
- **Definition**: The total recognized earned sales value realized from finalized transactions.
- **Formula**: `SUM(Quantity * Unit_price - Discount_amount)` where `Status = 'Completed'`.
- **Accounting Rule**: Returned merchandise (`Status = 'Returned'`) is strictly excluded from net revenue to adhere to GAAP and IFRS revenue-recognition standards.

---

### Gross Revenue
- **Definition**: The aggregate gross dollar or currency volume of orders before applying discounts or promotional credits.
- **Formula**: `SUM(Quantity * Unit_price)` where `Status = 'Completed'`.
- **Key Metric**: Total gross revenue across the 50 completed orders in the benchmark dataset is ₹14,49,700.00.

---

### Returned Orders
- **Definition**: Orders where merchandise was rejected, cancelled, or returned by the customer post-order placement.
- **Dataset Occurrence**: Identified by `Status = 'Returned'`. In this dataset, transaction `TXN0033` (2 units of Monitor in West region, grossing ₹19,000.00) represents the sole return.
- **Handling Policy**: Retained in SQLite for return rate auditing, but excluded from all net sales metrics.

---

### Average Order Value (AOV)
- **Definition**: The average net sales revenue generated per qualifying completed transaction.
- **Formula**: `AVG(net_revenue)` where `Status = 'Completed' AND Quantity > 0`.
- **Policy Exclusion**: Zero-quantity transaction `TXN0034` is strictly excluded from the divisor to prevent artificial deflation of the average basket size. Benchmark AOV is ₹29,493.88 across 49 qualifying orders.

---

### Units Sold & Quantity
- **Definition**: The total physical unit count of products ordered and fulfilled.
- **Formula**: `SUM(Quantity)` on completed sales.
- **Basket Sizing**: Average units per order is computed as `AVG(Quantity)` where `Quantity > 0` (averaging 3.5 units per order).

---

### Discount Amount
- **Definition**: Lump-sum promotional deduction applied at the transaction order level.
- **Impact**: In the dataset, four transactions have discounts applied (`TXN0004`: ₹500, `TXN0018`: ₹1,000, `TXN0027`: ₹2,500, `TXN0041`: ₹500), totaling ₹4,500.00 across the portfolio.

---

### Product Catalog
- **Allowed Products**:
  - `Laptop Stand`: Invariant catalog price of ₹1,200.00.
  - `Wireless Mouse`: Invariant catalog price of ₹800.00.
  - `Office Chair`: Invariant catalog price of ₹7,500.00.
  - `Monitor`: Invariant catalog price of ₹9,500.00. Top revenue generator.
  - `Keyboard`: Invariant catalog price of ₹900.00.
  - `Desk`: Invariant catalog price of ₹12,000.00.
  - `USB-C Hub`: Invariant catalog price of ₹1,500.00.

---

### Product Categories
- **Electronics**: Includes Monitor, USB-C Hub.
- **Furniture**: Includes Office Chair, Desk.
- **Accessories**: Includes Laptop Stand, Wireless Mouse, Keyboard.

---

### Geographic Regions
- **North**: Generated ₹3,84,450.00 in completed net revenue.
- **South**: Top revenue region with ₹4,11,200.00 in completed net revenue.
- **East**: Generated ₹3,52,700.00 in completed net revenue.
- **West**: Generated ₹2,96,850.00 in completed net revenue (excludes ₹19,000 returned order TXN0033).

---

### Sales Channels
- **Online**: Orders fulfilled via web commerce.
- **Retail**: Orders placed through physical brick-and-mortar storefronts.

---

### Data Handling & Imputation Rules
- **TXN0051 Imputation**: Row 52 had an unpopulated `Unit_price`. Imputed using the statistical mode (₹1,200.00) of all other `Laptop Stand` sales in the dataset, flagged with `is_imputed = 1`.
- **Zero-Quantity Handling**: Row 35 (`TXN0034`) has `Quantity = 0`. Retained in database with `is_imputed = 0`, but excluded from AOV calculations.
- **Date Standardization**: Order dates in format `DD-MM-YYYY` are standardized into `YYYY-MM-DD` and `YYYY-MM` month buckets.
