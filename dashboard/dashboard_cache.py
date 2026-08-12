"""Small helpers for Streamlit pages to use the dashboard cache."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_CACHE_DIR = Path("results/dashboard_cache/final_production_v1")


def load_manifest(cache_dir: str) -> dict:
    path = Path(cache_dir) / "cache_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache_csv(cache_dir: str, filename: str) -> pd.DataFrame:
    path = Path(cache_dir) / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def sidebar_cache_controls(key_prefix: str = "global") -> tuple[bool, Path, dict]:
    st.sidebar.header("Dashboard Cache")
    use_cache = st.sidebar.checkbox("Use dashboard cache", value=True, key=f"{key_prefix}_use_dashboard_cache")
    source_mode = st.sidebar.radio(
        "Source mode",
        ["final_production_v1", "historical/debug"],
        index=0,
        key=f"{key_prefix}_source_mode",
        help="Final production is the meeting default; historical/debug exposes older smoke/medium paths.",
    )
    default_cache = DEFAULT_CACHE_DIR if source_mode == "final_production_v1" else Path("results/dashboard_cache")
    cache_path = Path(
        st.sidebar.text_input(
            "Dashboard cache path",
            value=str(default_cache),
            key=f"{key_prefix}_dashboard_cache_path",
        )
    )
    return use_cache, cache_path, load_manifest(str(cache_path))


def show_cache_badge(use_cache: bool, cache_dir: Path, manifest: dict) -> None:
    if not use_cache:
        st.info("Dashboard cache disabled: page is using the normal interactive loaders.")
        return
    status = "ready" if manifest.get("dashboard_ready") else "partial" if manifest else "missing"
    raw_data_level = manifest.get("data_level", "missing")
    data_level = "preview" if raw_data_level == "smoke" else raw_data_level
    source_runset = manifest.get("source_runset", "unknown")
    created_at = manifest.get("created_at", "unavailable")
    cols = st.columns(3)
    cols[0].metric("Data level", data_level)
    cols[1].metric("Cache status", status)
    cols[2].metric("Last prepared", created_at)
    st.caption(f"Cache path: {cache_dir}; source runset: {source_runset}")
    if source_runset and source_runset != "final_production_v1":
        st.warning("Historical/debug source selected. Do not use this view for final meeting claims.")
    if raw_data_level == "smoke":
        st.warning("Preview cache only — do not use for final scientific conclusions.")
    if status != "ready":
        missing = manifest.get("files_missing", []) if manifest else ["cache_manifest.json"]
        if missing:
            st.error(f"Dashboard cache is {status}. Missing: {', '.join(map(str, missing))}. Cache path: {cache_dir}")
        else:
            st.info(f"Dashboard cache is {status}. This is expected for smoke/preview data. Cache path: {cache_dir}")
    cache_warnings = []
    model_caveats = []
    for warning in manifest.get("warnings", []) if manifest else []:
        if not warning:
            continue
        text = str(warning)
        if text.lower().startswith(("student ", "laplace ", "logistic ")):
            model_caveats.append(text)
        else:
            cache_warnings.append(text)
    for warning in cache_warnings:
        st.warning(warning)
    if model_caveats:
        with st.expander("Model caveats in this cache", expanded=False):
            for caveat in model_caveats:
                st.info(caveat)


def require_cache_file(cache_dir: Path, filename: str) -> Path | None:
    path = cache_dir / filename
    if path.exists():
        return path
    st.error(f"Missing dashboard cache file: {path}")
    return None
