"""
Reusable UI Metric Cards and Theme Components for the Smart Electricity Analytics Dashboard.
Styled with a Midnight Obsidian & Electric Indigo-Violet SaaS design system.
"""

import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    delta: str = None,
    delta_color: str = "normal",
    help_text: str = None
):
    """Renders a styled metric card."""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def render_top_navbar(active_section: str, dataset_label: str, source_tag: str = "London SmartMeter"):
    """Renders the capsule top navigation bar inspired by modern SaaS dark UI."""
    clean_section = active_section.split(" ", 1)[-1] if " " in active_section else active_section
    st.markdown(
        f"""
        <div class="seca-topbar">
            <div class="seca-topbar-left">
                <span class="seca-brand-mark">SECA</span>
                <span class="seca-topbar-divider"></span>
                <span class="seca-nav-pill active">{clean_section}</span>
                <span class="seca-nav-pill">{source_tag}</span>
                <span class="seca-nav-pill">PySpark 4.3</span>
            </div>
            <div class="seca-topbar-right">
                <span class="seca-status-chip">● {dataset_label}</span>
                <span class="seca-cta-pill">LIVE ANALYTICS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_header_banner(
    title: str,
    subtitle: str,
    badge: str = "London SmartMeter Dataset",
    right_tag: str = "UK POWER NETWORKS"
):
    """Renders an obsidian & electric indigo-violet hero banner with subtle ambient glow."""
    st.markdown(
        f"""
        <div class="seca-hero-card">
            <div class="seca-hero-inner">
                <div>
                    <div class="seca-hero-eyebrow">{badge}</div>
                    <h1 class="seca-hero-title">{title}</h1>
                    <p class="seca-hero-subtitle">{subtitle}</p>
                </div>
                <div class="seca-hero-badge">
                    <span class="seca-hero-dot"></span>
                    <span>{right_tag}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_strip(items: list):
    """
    Renders a row of glowing SaaS-style pill feature chips matching the reference grid.
    Each item is a tuple: (label, is_highlighted)
    """
    cols = st.columns(len(items))
    for col, (text, highlighted) in zip(cols, items):
        cls = "seca-chip seca-chip-active" if highlighted else "seca-chip"
        with col:
            st.markdown(
                f"""
                <div class="{cls}">
                    <span class="seca-chip-icon">✦</span>
                    <span class="seca-chip-text">{text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
