"""
Reusable UI Metric Cards for the Smart Electricity Analytics Dashboard.
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


def render_header_banner(title: str, subtitle: str, badge: str = "London SmartMeter Dataset"):
    """Renders a sleek energy analytics top banner."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0e1e38 0%, #17325c 100%); 
                    padding: 24px 28px; border-radius: 12px; border-left: 6px solid #00d2ff; 
                    margin-bottom: 24px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">
                        ⚡ {title}
                    </h1>
                    <p style="color: #a0c0e8; margin: 6px 0 0 0; font-size: 15px;">
                        {subtitle}
                    </p>
                </div>
                <div style="background: rgba(0, 210, 255, 0.15); border: 1px solid #00d2ff; 
                            color: #00d2ff; padding: 6px 14px; border-radius: 20px; 
                            font-size: 12px; font-weight: 600; text-transform: uppercase;">
                    {badge}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
