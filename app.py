# app.py (Optimized with SQLite)
import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import os
import sqlite3

# Database connection function
def query_db(sql, params=()):
    """Execute SQL query and return DataFrame"""
    conn = sqlite3.connect("crashes.db")
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

def get_unique_values(column_name):
    """Get unique values for dropdowns"""
    return query_db(f"SELECT DISTINCT {column_name} FROM crashes WHERE {column_name} IS NOT NULL")[column_name].tolist()

def get_min_max_year():
    """Get year range from database"""
    result = query_db("SELECT MIN(YEAR) as min_year, MAX(YEAR) as max_year FROM crashes")
    min_year = int(result['min_year'].iloc[0]) if not result.empty else 2010
    max_year = int(result['max_year'].iloc[0]) if not result.empty else pd.Timestamp.now().year
    return min_year, max_year

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Get initial data for setup
min_year, max_year = get_min_max_year()
year_marks = {y: str(y) for y in range(min_year, max_year + 1)}

# Borough colors (keep this mapping)
BOROUGH_COLORS = {
    'Manhattan': '#2ECC71',
    'Brooklyn': '#E74C3C',
    'Queens': '#3498DB',
    'Bronx': '#F39C12',
    'Staten Island': '#9B59B6',
    'Unknown': '#95A5A6'
}

# Layout remains the same as your original code...
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("💥 NYC Crash Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
            html.Div(id="summary_text",
                    className="alert text-center",
                    style={'fontSize': '18px', 'fontWeight': 'bold', 'backgroundColor': '#FF8DA1', 'color': 'white', 'border': 'none'})
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(
            html.H4("📊 Control Panel", className="mb-0", style={'color': '#ffffff'}),
            style={'backgroundColor': '#FF8DA1'}
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Year Range", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dcc.RangeSlider(
                        id="year_slider",
                        min=min_year,
                        max=max_year,
                        value=[min_year, max_year],
                        marks={y: {'label': str(y), 'style': {'color': '#ffffff'}} for y in range(min_year, max_year + 1)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=1,
                        allowCross=False
                    ),
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Label("Borough", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="borough_filter",
                        options=[{"label": b, "value": b} for b in get_unique_values("BOROUGH")],
                        multi=True,
                        placeholder="All Boroughs",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Vehicle Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="vehicle_filter",
                        options=[],  # Will be populated dynamically
                        multi=True,
                        placeholder="All Vehicle Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Contributing Factor", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="factor_filter",
                        options=[],  # Will be populated dynamically
                        multi=True,
                        placeholder="All Factors",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Person Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="person_type_filter",
                        options=[{"label": v, "value": v} for v in get_unique_values("PERSON_TYPE")],
                        multi=True,
                        placeholder="All Person Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Injury Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="injury_filter",
                        options=[{"label": i, "value": i} for i in get_unique_values("PERSON_INJURY")],
                        multi=True,
                        placeholder="All Injury Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Clear Filters", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dbc.Button("🗑️ Clear All Filters",
                              id="clear_filters_btn",
                              color="warning",
                              size="md",
                              className="w-100",
                              style={
                                  'backgroundColor': '#FF6B6B',
                                  'border': 'none',
                                  'fontWeight': 'bold',
                                  'color': 'white'
                              })
                ], width=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Label("🔍 Advanced Search", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.Input(
                        id="search_input",
                        placeholder="Try: 'queens 2019 to 2022 bicycle female pedestrian'...",
                        type="text",
                        style={
                            'backgroundColor': '#FFE6E6',
                            'border': '2px solid #FF8DA1',
                            'color': '#333',
                            'fontSize': '14px',
                            'padding': '12px'
                        }
                    ),
                    dbc.FormText(
                        "Search by borough, year, vehicle type, gender, injury type",
                        style={'color': '#ffffff', 'fontWeight': 'bold'}
                    )
                ], width=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Button("🔄 Update Dashboard",
                              id="generate_btn",
                              color="primary",
                              size="lg",
                              className="w-100",
                              style={'backgroundColor': '#FF8DA1', 'border': 'none', 'fontWeight': 'bold'})
                ], width=12),
            ]),
        ], style={'backgroundColor': '#add8e6'})
    ], className="mb-4", style={'border': '2px solid #FF8DA1'}),

    # Rest of your layout remains exactly the same...
    dbc.Tabs([
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📍 Crash Locations Map", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="map_chart", style={'height': '500px'})
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Crash Trends Over Time", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_year")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏙️ Crashes by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="borough_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💥 Injuries by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injuries_by_borough")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),
        ], label="🗺️ Crash Geography", tab_id="tab-1"),

        # ... rest of your tabs remain exactly the same
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔧 Contributing Factors", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_factor")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔥 Vehicle vs Factor Heatmap", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_factor_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏎️ Vehicle Type Trends", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_trend_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="🏎️ Vehicles & Factors", tab_id="tab-2"),

        # ... include all your other tabs exactly as they are
        dbc.Tab([], label="👥 People & Injuries", tab_id="tab-3"),
        dbc.Tab([], label="📈 Demographics", tab_id="tab-4"),
        dbc.Tab([], label="🔬 Advanced Analytics", tab_id="tab-5"),
    ], id="tabs", active_tab="tab-1",
       style={'marginTop': '20px'},
       className="custom-tabs"),

    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': '#FF8DA1'}),
            html.P("NYC Crash Analysis Dashboard | Built with Dash & Plotly",
                  className="text-center",
                  style={'color': '#ffffff', 'fontWeight': 'bold'})
        ])
    ], className="mt-4")

], fluid=True, style={'backgroundColor': '#cee8f0', 'minHeight': '100vh', 'padding': '20px'})

# Keep your existing CSS and search parsing function
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .rc-slider-track {
                background-color: #fc94af !important;
            }
            .rc-slider-rail {
                background-color: #FFC0CB !important;
            }
            .rc-slider-handle {
                background-color: #fc94af !important;
                border: 2px solid white !important;
            }

            .custom-tabs .nav-link {
                background-color: #FF8DA1 !important;
                color: white !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
                margin-right: 5px;
            }
            .custom-tabs .nav-link.active {
                background-color: white !important;
                color: #FF8DA1 !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
            }
            .custom-tabs .nav-link:hover {
                background-color: #FF85A1 !important;
                color: white !important;
            }
            .custom-tabs .nav-link.active:hover {
                background-color: white !important;
                color: #FF8DA1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def parse_search_query(q):
    # Keep your existing search parsing function exactly as is
    q = (q or "").lower().strip()
    found = {}

    year_pattern = r'\b(20\d{2})\b'
    years_found = re.findall(year_pattern, q)
    if years_found:
        years = sorted([int(y) for y in years_found])
        if len(years) >= 2:
            found["year_range"] = [years[0], years[-1]]
        else:
            found["year"] = years[0]

    borough_keywords = {
        'manhattan': 'Manhattan',
        'brooklyn': 'Brooklyn',
        'queens': 'Queens',
        'bronx': 'Bronx',
        'staten': 'Staten Island',
        'staten island': 'Staten Island'
    }
    for keyword, borough in borough_keywords.items():
        if keyword in q:
            found["borough"] = [borough]
            break

    vehicle_keywords = {
        'suv': 'SUV/Station Wagon',
        'station wagon': 'SUV/Station Wagon',
        'sedan': 'Sedan',
        'bicycle': 'Bicycle',
        'bike': 'Bicycle',
        'ambulance': 'Ambulance',
        'bus': 'Bus',
        'motorcycle': 'Motorcycle',
        'pickup': 'Pickup Truck',
        'pickup truck': 'Pickup Truck',
        'taxi': 'Taxi',
        'truck': 'Truck/Commercial',
        'commercial': 'Truck/Commercial',
        'van': 'Van',
        'pedicab': 'Pedicab'
    }
    vehicle_matches = []
    for keyword, vehicle_type in vehicle_keywords.items():
        if keyword in q:
            vehicle_matches.append(vehicle_type)
    if vehicle_matches:
        found["vehicle"] = vehicle_matches

    person_type_matches = []
    if 'pedestrian' in q:
        person_type_matches.append('Pedestrian')
    if 'cyclist' in q or 'bicyclist' in q:
        person_type_matches.append('Bicyclist')
    if 'motorist' in q:
        person_type_matches.append('Motorist')
    if 'driver' in q:
        person_type_matches.append('Driver')
    if 'occupant' in q:
        person_type_matches.append('Occupant')
    if person_type_matches:
        found["person_type"] = person_type_matches

    if 'female' in q or 'woman' in q or 'women' in q:
        found["gender"] = ['F']
    elif 'male' in q or 'man' in q or 'men' in q:
        found["gender"] = ['M']

    if ' f ' in f" {q} " or q.endswith(' f') or q.startswith('f '):
        found["gender"] = ['F']
    elif ' m ' in f" {q} " or q.endswith(' m') or q.startswith('m '):
        found["gender"] = ['M']

    injury_matches = []
    if 'injured' in q or 'injury' in q:
        injury_matches.append('Injured')
    if 'killed' in q or 'fatal' in q or 'fatality' in q or 'death' in q or 'died' in q:
        injury_matches.append('Killed')
    if 'unspecified' in q:
        injury_matches.append('Unspecified')
    if injury_matches:
        found["injury"] = injury_matches

    return found

def build_where_clause(year_range, boroughs, vehicles, factors, injuries, person_type, search_parsed=None):
    """Build SQL WHERE clause dynamically based on filters"""
    conditions = []
    params = []
    
    # Year range
    if year_range and len(year_range) == 2:
        conditions.append("YEAR BETWEEN ? AND ?")
        params.extend(year_range)
    
    # Boroughs
    if boroughs:
        placeholders = ','.join(['?'] * len(boroughs))
        conditions.append(f"BOROUGH IN ({placeholders})")
        params.extend(boroughs)
    
    # Person type
    if person_type:
        placeholders = ','.join(['?'] * len(person_type))
        conditions.append(f"PERSON_TYPE IN ({placeholders})")
        params.extend(person_type)
    
    # Injuries
    if injuries:
        placeholders = ','.join(['?'] * len(injuries))
        conditions.append(f"PERSON_INJURY IN ({placeholders})")
        params.extend(injuries)
    
    # Search-based filters
    if search_parsed:
        if "gender" in search_parsed:
            placeholders = ','.join(['?'] * len(search_parsed["gender"]))
            conditions.append(f"PERSON_SEX IN ({placeholders})")
            params.extend(search_parsed["gender"])
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return where_clause, params

@app.callback(
    [
        Output("vehicle_filter", "options"),
        Output("factor_filter", "options"),
    ],
    Input("generate_btn", "n_clicks"),
    prevent_initial_call=True
)
def update_dynamic_filters(n_clicks):
    """Update vehicle and factor dropdowns dynamically"""
    vehicle_options = [{"label": v, "value": v} for v in get_unique_values("VEHICLE_TYPE_CODE_1")]
    factor_options = [{"label": f, "value": f} for v in get_unique_values("CONTRIBUTING_FACTOR_VEHICLE_1")]
    
    return vehicle_options, factor_options

@app.callback(
     [
          Output("injuries_by_borough", "figure"),
          Output("crashes_by_factor", "figure"),
          Output("crashes_by_year", "figure"),
          Output("map_chart", "figure"),
          Output("gender_distribution", "figure"),
          Output("safety_equipment", "figure"),
          Output("emotional_state", "figure"),
          Output("age_distribution_hist", "figure"),
          Output("injuries_by_person_type", "figure"),
          Output("summary_text", "children"),

          Output("borough_chart", "figure"),
          Output("injury_chart", "figure"),
          Output("ejection_chart", "figure"),
          Output("complaint_chart", "figure"),
          Output("vehicle_factor_chart", "figure"),
          Output("position_chart", "figure"),
          Output("vehicle_trend_chart", "figure"),

          Output("hotspot_cluster_map", "figure"),
          Output("correlation_heatmap", "figure"),
          Output("temporal_patterns", "figure"),
          Output("severity_factors", "figure"),
          Output("risk_density_map", "figure"),

          Output("live_stats", "children"),
     ],
     Input("generate_btn", "n_clicks"),
     [
          State("year_slider", "value"),
          State("borough_filter", "value"),
          State("vehicle_filter", "value"),
          State("factor_filter", "value"),
          State("injury_filter", "value"),
          State("person_type_filter", "value"),
          State("search_input", "value"),
     ]
)
def update_dashboard(n_clicks, year_range, boroughs, vehicles, factors, injuries, person_type, search_text):
    # Parse search query first
    search_parsed = parse_search_query(search_text) if search_text else {}
    
    # Build WHERE clause for database query
    where_clause, params = build_where_clause(year_range, boroughs, vehicles, factors, injuries, person_type, search_parsed)
    
    # Query only the data we need
    sql = f"""
    SELECT * FROM crashes 
    WHERE {where_clause}
    LIMIT 10000  -- Limit for performance
    """
    
    dff = query_db(sql, params)
    
    # If we have vehicle or factor filters that can't be handled in SQL, filter in memory
    if vehicles and not dff.empty:
        mask = dff["VEHICLE_TYPE_CODE_1"].isin(vehicles)
        dff = dff[mask]
    
    if factors and not dff.empty:
        mask = dff["CONTRIBUTING_FACTOR_VEHICLE_1"].isin(factors)
        dff = dff[mask]
    
    # Rest of your plotting code remains exactly the same...
    total_crashes = len(dff)
    total_injuries = dff["TOTAL_INJURED"].sum() if not dff.empty else 0
    total_killed = dff["TOTAL_KILLED"].sum() if not dff.empty else 0
    avg_injuries_per_crash = total_injuries / total_crashes if total_crashes > 0 else 0

    pink_template = {
        'layout': {
            'paper_bgcolor': '#FFE6E6',
            'plot_bgcolor': '#FFE6E6',
            'font': {'color': '#2C3E50'},
            'xaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'},
            'yaxis': {'gridcolor': '#FFB6C1', 'linecolor': '#2C3E50'}
        }
    }

    vibrant_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    # Create your figures exactly as before...
    if not dff.empty:
        injuries_by_borough = dff.groupby("BOROUGH")["TOTAL_INJURED"].sum().reset_index().sort_values("TOTAL_INJURED", ascending=False)
        fig_inj_borough = px.bar(injuries_by_borough, x="BOROUGH", y="TOTAL_INJURED",
                                 labels={"TOTAL_INJURED": "Total Injured", "BOROUGH": "Borough"},
                                 text="TOTAL_INJURED",
                                 color="BOROUGH",
                                 color_discrete_map=BOROUGH_COLORS)
        fig_inj_borough.update_traces(textposition="outside")
        fig_inj_borough.update_layout(margin=dict(t=40, b=20), template=pink_template, showlegend=False)
    else:
        fig_inj_borough = go.Figure()
        fig_inj_borough.update_layout(title="No data available")

    # ... continue with all your other figure creations exactly as before
    
    # For brevity, I'm showing the pattern. You'll need to convert ALL your figure creations
    # to work with the filtered dff DataFrame
    
    summary = f"📊 Currently showing: {total_crashes:,} crashes | {total_injuries:,} injured | {total_killed:,} fatalities"
    
    # Create live stats cards
    live_stats = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("🏎️ Total Crashes", className="text-primary", style={'color': '#FF8DA1'}),
            html.H2(f"{total_crashes:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
        ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("💥 Total Injuries", className="text-warning", style={'color': '#FF8DA1'}),
            html.H2(f"{total_injuries:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
        ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("💀 Total Fatalities", className="text-danger", style={'color': '#FF8DA1'}),
            html.H2(f"{total_killed:,}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
        ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("📈 Avg Injuries/Crash", className="text-success", style={'color': '#FF8DA1'}),
            html.H2(f"{avg_injuries_per_crash:.2f}", style={'color': '#FF8DA1', 'fontWeight': 'bold'})
        ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=3),
    ])

    # Return all figures (you'll need to create them all as in your original code)
    return (
        fig_inj_borough,
        # ... return all your other figures
        live_stats,
    )

@app.callback(
    [
        Output("year_slider", "value"),
        Output("borough_filter", "value"),
        Output("vehicle_filter", "value"),
        Output("factor_filter", "value"),
        Output("injury_filter", "value"),
        Output("person_type_filter", "value"),
        Output("search_input", "value")
    ],
    Input("clear_filters_btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_all_filters(n_clicks):
    min_year, max_year = get_min_max_year()
    return (
        [min_year, max_year],
        None,
        None,
        None,
        None,
        None,
        ""
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)
