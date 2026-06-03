"""Small helpers for Streamlit pages to use the dashboard cache."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_CACHE_DIR = Path("results/dashboard_cache")


@st.cache_data(show_spinner=False)
def load_manifest(cache_dir: str) -> dict:
    path = Path(cache_dir) / "cache_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_cache_csv(cache_dir: str, filename: str) -> pd.DataFrame:
    path = Path(cache_dir) / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def sidebar_cache_controls(key_prefix: str = "global") -> tuple[bool, Path, dict]:
    st.sidebar.header("Dashboard Cache")
    use_cache = st.sidebar.checkbox("Use dashboard cache", value=True, key=f"{key_prefix}_use_dashboard_cache")
    cache_path = Path(
        st.sidebar.text_input(
            "Dashboard cache path",
            value=str(DEFAULT_CACHE_DIR),
            key=f"{key_prefix}_dashboard_cache_path",
        )
    )
    return use_cache, cache_path, load_manifest(str(cache_path))


def show_cache_badge(use_cache: bool, cache_dir: Path, manifest: dict) -> None:
    if not use_cache:
        st.info("Dashboard cache disabled: page is using the normal interactive loaders.")
        return
    status = "ready" if manifest.get("dashboard_ready") else "partial" if manifest else "missing"
    data_level = manifest.get("data_level", "missing")
    created_at = manifest.get("created_at", "unavailable")
    caveat = "Student k=1,n=10 unresolved"
    cols = st.columns(4)
    cols[0].metric("Data level", data_level)
    cols[1].metric("Cache status", status)
    cols[2].metric("Last prepared", created_at)
    cols[3].metric("Major caveat", caveat)
    if status != "ready":
        missing = manifest.get("files_missing", []) if manifest else ["cache_manifest.json"]
        st.error(f"Dashboard cache is {status}. Missing: {', '.join(map(str, missing))}. Cache path: {cache_dir}")


def require_cache_file(cache_dir: Path, filename: str) -> Path | None:
    path = cache_dir / filename
    if path.exists():
        return path
    st.error(f"Missing dashboard cache file: {path}")
    return None
