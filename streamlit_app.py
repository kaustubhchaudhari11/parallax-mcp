"""PARALLAX release dashboard.

Run from the repository root:

    uv run --extra dashboard streamlit run streamlit_app.py

The process does not read an API key and does not call a model.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from parallax.dashboard.components import inject_css
from parallax.dashboard.data import (
    DashboardLoadError,
    Disclosure,
    ReleaseBundle,
    disclosure_for,
    load_bundled,
    local_runs_dir,
    publish_runs,
    warnings_for_display,
)
from parallax.dashboard.views import (
    render_page,
    render_page_picker,
    render_sidebar_brand,
    render_source_picker,
)


def main() -> None:
    """Start the read-only dashboard."""

    st.set_page_config(
        page_title="PARALLAX",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    with st.sidebar:
        render_sidebar_brand()
        source_key = render_source_picker(local_runs_dir() is not None)
    bundle, warnings, source_label, disclosure = _load_source(source_key)
    if bundle is None:
        return
    with st.sidebar:
        page = render_page_picker()
    for warning in warnings_for_display(warnings, disclosure):
        st.warning(warning)
    render_page(page, bundle, source_label, disclosure)


def _load_source(source_key: str) -> tuple[ReleaseBundle | None, list[str], str, Disclosure]:
    if source_key == "local":
        directory = local_runs_dir()
        if directory is None:
            return _bundled_or_stop(["Local runs directory is absent. Showing the bundled release."])
        try:
            published = publish_runs(directory, strict=False)
        except DashboardLoadError as exc:
            st.error(str(exc))
            return None, [], "", disclosure_for("bundled")
        readable = published.bundle.experiments or published.bundle.xray or published.bundle.controls
        if not readable:
            notes = [
                "Local directory had no readable result summary. Showing the bundled release.",
                *published.warnings,
            ]
            return _bundled_or_stop(notes)
        return published.bundle, list(published.warnings), "Local runs directory", disclosure_for("local")
    return _bundled_or_stop([])


def _bundled_or_stop(notes: list[str]) -> tuple[ReleaseBundle | None, list[str], str, Disclosure]:
    bundled, error = _bundled()
    if bundled is None:
        st.error(error)
        return None, [], "", disclosure_for("bundled")
    return bundled, notes, "Bundled release", disclosure_for("bundled")


def _bundled() -> tuple[object, str]:
    try:
        return load_bundled(), ""
    except DashboardLoadError as exc:
        return None, f"The bundled release could not be loaded. {exc}"


if __name__ == "__main__":
    main()
