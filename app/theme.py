# ── AttritionIQ Dashboard Theme ───────────────────────────────────────────────
# Modern Minimal + Emerald Green
# Edit this file to change colors across the entire dashboard

# Core accent
PRIMARY       = "#00C853"
PRIMARY_DARK  = "#00897B"
PRIMARY_LIGHT = "#E8F5E9"

# Risk colors
HIGH_RISK     = "#FF5252"
MEDIUM_RISK   = "#FFB300"
LOW_RISK      = "#00C853"

# Chart colors
CHART_MAIN    = "#00897B"
CHART_SEC     = "#26C6DA"
CHART_THIRD   = "#5C6BC0"
CHART_FOURTH  = "#EF5350"

# Neutrals
BG            = "#FFFFFF"
BG_SUBTLE     = "#F9FAFB"
BORDER        = "#EEEEEE"
TEXT_DARK     = "#1A1A2E"
TEXT_MID      = "#546E7A"
TEXT_LIGHT    = "#90A4AE"

# Plotly base layout — safe version with NO legend/axis keys
# These are applied to every chart for consistency
PLOTLY_BASE = dict(
    paper_bgcolor = "#FFFFFF",
    plot_bgcolor  = "#FFFFFF",
    font          = dict(family="Arial", color="#546E7A", size=11),
    title_font    = dict(family="Arial", color="#1A1A2E", size=13),
    margin        = dict(l=20, r=20, t=40, b=20),
)

# Safe axis style — apply individually per chart
AXIS_STYLE = dict(
    gridcolor  = "#F5F5F5",
    linecolor  = "#EEEEEE",
    tickfont   = dict(size=10, color="#90A4AE"),
    showgrid   = True,
    zeroline   = False,
)

# Color map for risk labels
RISK_COLOR_MAP = {
    "High"   : "#FF5252",
    "Medium" : "#FFB300",
    "Low"    : "#00C853",
}

# Sequential scale for heatmaps/gradients
GREEN_SEQ = [
    [0.0, "#E8F5E9"],
    [0.3, "#81C784"],
    [0.6, "#00C853"],
    [1.0, "#00600F"],
]