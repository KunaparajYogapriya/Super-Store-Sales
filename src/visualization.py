"""
Custom SaaS-Style Plotly Visualizations Module for Superstore Sales Intelligence.
"""

from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Design System Palette
PALETTE = {
    'primary': '#1E293B',
    'accent': '#2563EB',
    'light_blue': '#38BDF8',
    'emerald': '#10B981',
    'amber': '#F59E0B',
    'crimson': '#EF4444',
    'neutral_light': '#F8FAFC',
    'neutral_dark': '#0F172A',
    'grid_color': '#E2E8F0'
}

PLOTLY_TEMPLATE = "plotly_white"


def apply_custom_layout(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply unified SaaS visual styling to Plotly figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=18, color=PALETTE['neutral_dark'], family="Inter, sans-serif"),
            x=0.0,
            xanchor='left'
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=PALETTE['grid_color'],
            zeroline=False,
            titlefont=dict(size=13, color="#475569")
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=PALETTE['grid_color'],
            zeroline=False,
            titlefont=dict(size=13, color="#475569")
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", color="#334155"),
        hoverlabel=dict(bgcolor="#0F172A", font_size=13, font_color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_sales_profit_trend(df: pd.DataFrame) -> go.Figure:
    """Plot monthly trend of Sales and Profit."""
    df_copy = df.copy()
    df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])
    monthly = df_copy.set_index('Order Date').resample('MS')[['Sales', 'Profit']].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['Order Date'],
        y=monthly['Sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color=PALETTE['accent'], width=3),
        marker=dict(size=6),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Sales:</b> $%{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=monthly['Order Date'],
        y=monthly['Profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color=PALETTE['emerald'], width=3),
        marker=dict(size=6),
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Profit:</b> $%{y:,.2f}<extra></extra>"
    ))
    
    return apply_custom_layout(fig, title="Monthly Sales & Profit Trend", x_title="Order Date", y_title="Amount ($)")


def plot_category_breakdown(df: pd.DataFrame, metric: str = "Sales") -> go.Figure:
    """Horizontal bar chart for Category breakdown."""
    cat_df = df.groupby('Category')[metric].sum().reset_index().sort_values(metric, ascending=True)
    color = PALETTE['accent'] if metric == "Sales" else PALETTE['emerald']
    
    fig = px.bar(
        cat_df,
        y='Category',
        x=metric,
        orientation='h',
        text_auto='.2s',
        color_discrete_sequence=[color]
    )
    fig.update_traces(textposition='outside', marker_line_color=PALETTE['primary'], marker_line_width=1)
    return apply_custom_layout(fig, title=f"Total {metric} by Category", x_title=f"{metric} ($)", y_title="Category")


def plot_subcategory_performance(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart for Sub-Category Sales & Profit."""
    sub_df = df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index().sort_values('Sales', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub_df['Sub-Category'],
        y=sub_df['Sales'],
        name='Sales',
        marker_color=PALETTE['accent']
    ))
    fig.add_trace(go.Bar(
        x=sub_df['Sub-Category'],
        y=sub_df['Profit'],
        name='Profit',
        marker_color=np.where(sub_df['Profit'] >= 0, PALETTE['emerald'], PALETTE['crimson'])
    ))
    
    fig.update_layout(barmode='group')
    return apply_custom_layout(fig, title="Sales & Profit by Sub-Category", x_title="Sub-Category", y_title="Amount ($)")


def plot_discount_vs_profit(df: pd.DataFrame) -> go.Figure:
    """Scatter plot and box plot for Discount vs Profit Margin."""
    df_copy = df.copy()
    df_copy['Discount Rate'] = (df_copy['Discount'] * 100).astype(str) + '%'
    disc_df = df_copy.groupby('Discount')[['Profit', 'Sales']].agg({'Profit': ['sum', 'mean'], 'Sales': 'sum'}).reset_index()
    disc_df.columns = ['Discount', 'Total Profit', 'Avg Profit per Order', 'Total Sales']
    
    fig = px.scatter(
        df_copy.sample(min(1500, len(df_copy)), random_state=42),
        x='Discount',
        y='Profit',
        color='Category',
        size='Sales',
        hover_data=['Sub-Category', 'State'],
        color_discrete_sequence=[PALETTE['accent'], PALETTE['emerald'], PALETTE['amber']],
        opacity=0.7
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE['crimson'], annotation_text="Break-even Profit")
    return apply_custom_layout(fig, title="Discount Impact on Order Profitability", x_title="Discount Rate", y_title="Profit ($)")


def plot_regional_segment_matrix(df: pd.DataFrame) -> go.Figure:
    """Heatmap of Sales by Region and Segment."""
    matrix = df.pivot_table(index='Region', columns='Segment', values='Sales', aggfunc='sum')
    
    fig = px.imshow(
        matrix,
        labels=dict(x="Customer Segment", y="Region", color="Sales ($)"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Blues",
        text_auto='.3s'
    )
    return apply_custom_layout(fig, title="Sales Distribution Matrix (Region vs Segment)")


def plot_state_choropleth(df: pd.DataFrame, metric: str = "Sales") -> go.Figure:
    """US Map of Sales/Profit by State."""
    state_df = df.groupby('State')[metric].sum().reset_index()
    
    # Simple state code mapping dictionary
    us_state_to_abbrev = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
        "District of Columbia": "DC"
    }
    state_df['State Code'] = state_df['State'].map(us_state_to_abbrev)
    
    fig = px.choropleth(
        state_df,
        locations='State Code',
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale="Viridis" if metric == "Sales" else "RdYlGn",
        labels={metric: f"Total {metric} ($)"}
    )
    return apply_custom_layout(fig, title=f"US Geographic {metric} Map")


def plot_forecasting_chart(historical_ts: pd.Series, forecast_df: pd.DataFrame) -> go.Figure:
    """Plot monthly historical sales along with future forecast horizon and 80% confidence interval."""
    fig = go.Figure()
    
    # Historical Sales
    fig.add_trace(go.Scatter(
        x=historical_ts.index,
        y=historical_ts.values,
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color=PALETTE['primary'], width=2.5),
        marker=dict(size=4)
    ))
    
    # Forecast Sales
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['forecast_sales'],
        mode='lines+markers',
        name='Projected Forecast',
        line=dict(color=PALETTE['accent'], width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Confidence Interval Shading
    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(37, 99, 235, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='80% Confidence Interval'
        ))
        
    return apply_custom_layout(fig, title="Monthly Sales Forecast Horizon", x_title="Date", y_title="Sales ($)")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs Predicted") -> go.Figure:
    """Regression scatter plot comparing Actual vs Predicted values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(color=PALETTE['accent'], opacity=0.6, size=6),
        name='Predictions'
    ))
    
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=PALETTE['crimson'], dash='dash', width=2),
        name='Ideal (45° line)'
    ))
    
    return apply_custom_layout(fig, title=title, x_title="Actual Values", y_title="Predicted Values")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plot residual distribution histogram."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.histogram(
        residuals,
        nbins=40,
        color_discrete_sequence=[PALETTE['accent']],
        labels={'value': 'Residual (Actual - Predicted)'}
    )
    return apply_custom_layout(fig, title="Model Residual Error Distribution", x_title="Residual ($)", y_title="Count")
