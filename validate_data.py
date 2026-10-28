"""Validate the required Part A cleaning, storage and SQL checkpoints."""
from pathlib import Path
import re
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main() -> None:
    customers = pd.read_csv(ROOT / 'data/clean/customers.csv')
    products = pd.read_csv(ROOT / 'data/clean/products.csv')
    orders = pd.read_csv(ROOT / 'data/clean/orders.csv')
    checks = []
    checks.append(len(customers) == 1200)
    checks.append(len(products) == 40)
    checks.append(len(orders) == 12309 and orders.order_id.is_unique)
    checks.append(orders.status.value_counts().to_dict() == {'completed': 11110, 'cancelled': 1199})
    checks.append(orders.order_amount.notna().all())
    checks.append(orders.loc[orders.status.eq('completed'), 'delivery_minutes'].notna().all())
    checks.append(orders.loc[orders.status.eq('cancelled'), 'delivery_minutes'].isna().all())
    checks.append(round(orders.loc[orders.status.eq('completed'), 'order_amount'].sum(), 2) == 2998668.80)
    checks.append(customers.age.notna().all() and customers.age.between(10, 100).all())
    checks.append(set(customers.city) == {'Pune', 'Delhi', 'Bengaluru', 'Chennai', 'Hyderabad', 'Mumbai'})
    checks.append(set(customers.gender) <= {'male', 'female', 'unknown'})

    db_path = ROOT / 'data/instamart.db'
    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        checks.append({'customers', 'products', 'orders'} <= tables)
        sql_text = (ROOT / 'sql/queries.sql').read_text(encoding='utf-8')
        queries = re.findall(r'-- Q\d+\n(.*?);', sql_text, re.S)
        checks.append(len(queries) == 7)
        results = [pd.read_sql_query(q, db) for q in queries]

    checks.append(len(results[0]) == 30)
    checks.append(len(results[1]) == 30 and round(results[1].revenue.sum(), 2) == 2998668.80)
    top_cell = results[1].sort_values('revenue', ascending=False).iloc[0]
    checks.append(top_cell.city == 'Bengaluru' and top_cell.category == 'Electronics' and round(top_cell.revenue, 2) == 233909.45)
    checks.append(len(results[2]) == 11)
    checks.append(len(results[3]) == 25)
    checks.append(len(results[4]) == 550)
    checks.append(len(results[5]) == 1175 and int((results[5].rnk != results[5].row_num).sum()) == 8)
    checks.append(len(results[6]) == 183 and round(float(results[6].cumulative_revenue.iloc[-1]), 2) == 2998668.80)
    if not all(checks):
        failed = [str(i + 1) for i, passed in enumerate(checks) if not passed]
        raise AssertionError('Validation checkpoints failed: ' + ', '.join(failed))
    print('ALL CHECKPOINTS PASSED')

if __name__ == '__main__':
    main()
