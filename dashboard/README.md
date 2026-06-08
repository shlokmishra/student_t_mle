# Dashboard

This folder contains the Streamlit dashboard and the scripts that build/check its cache.

Run locally:

```bash
streamlit run dashboard/app.py
```

Useful scripts:

- `dashboard/scripts/prepare_dashboard_cache.py` builds `results/dashboard_cache/`.
- `dashboard/scripts/check_dashboard_cache.py` verifies cache files and page paths.

Dashboard pages should import shared cache helpers from `dashboard.dashboard_cache`.
