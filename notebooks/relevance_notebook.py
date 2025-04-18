import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import re
    from datetime import datetime


    def parse_log_data(log_text):
        # Initialize data structures
        script_data = {}
        total_relevant = 0
        total_irrelevant = 0
        running_scripts = []
        current_script = None

        # Parse log line by line
        for line in log_text.split('\n'):
            # Check for script being processed
            if "Processing front page:" in line:
                current_script = line.split("Processing front page:")[1].strip()
                running_scripts.append(current_script)
                script_data[current_script] = {
                    'relevant': 0,
                    'irrelevant': 0,
                    'total': 0,
                    'status': 'started',
                    'error': None,
                    'vector_errors': 0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

            # Track relevant articles
            elif "Relevant article:" in line and current_script:
                script_data[current_script]['relevant'] += 1
                total_relevant += 1

            # Track non-relevant articles
            elif "Not relevant article:" in line and current_script:
                script_data[current_script]['irrelevant'] += 1
                total_irrelevant += 1

            # Track summary statistics
            elif "Relevant articles:" in line and current_script:
                match = re.search(r'Relevant articles: (\d+), Not relevant articles: (\d+)', line)
                if match:
                    relevant_count = int(match.group(1))
                    irrelevant_count = int(match.group(2))
                    script_data[current_script]['relevant'] = relevant_count
                    script_data[current_script]['irrelevant'] = irrelevant_count
                    script_data[current_script]['total'] = relevant_count + irrelevant_count

            # Track completion status
            elif "Completed processing for front page:" in line and current_script:
                script_data[current_script]['status'] = 'completed'

            # Track errors
            elif "ERROR:" in line and current_script:
                script_data[current_script]['status'] = 'failed'
                script_data[current_script]['error'] = line.split("ERROR:root:")[1].strip() if "ERROR:root:" in line else line

            # Track warnings and failures
            elif "WARNING:" in line and "No data found for" in line and current_script:
                script_data[current_script]['status'] = 'no_data'

            # Track vector errors
            elif "No vector found for content" in line and current_script:
                if 'vector_errors' not in script_data[current_script]:
                    script_data[current_script]['vector_errors'] = 0
                script_data[current_script]['vector_errors'] += 1

        return {
            'scripts': script_data,
            'total_relevant': total_relevant,
            'total_irrelevant': total_irrelevant,
            'running_scripts': running_scripts
        }
    return datetime, parse_log_data, re


@app.cell
def _():
    import pandas as pd
    import dash
    from dash import dcc, html
    import plotly.express as px
    from dash.dependencies import Input, Output

    def create_dashboard(data):
        # Convert data to DataFrames for easier manipulation
        scripts_df = pd.DataFrame([
            {
                'script': script,
                'relevant': info['relevant'],
                'irrelevant': info['irrelevant'],
                'total': info['total'],
                'status': info['status'],
                'vector_errors': info.get('vector_errors', 0),
                'relevance_ratio': info['relevant'] / info['total'] if info['total'] > 0 else 0
            }
            for script, info in data['scripts'].items()
        ])

        # Calculate status counts
        status_counts = scripts_df['status'].value_counts().to_dict()
        for status in ['completed', 'failed', 'no_data', 'started']:
            if status not in status_counts:
                status_counts[status] = 0

        # Calculate relevance ratio safely
        total_articles = data['total_relevant'] + data['total_irrelevant']
        relevance_ratio = (data['total_relevant'] / total_articles * 100) if total_articles > 0 else 0

        # Sort scripts by total articles
        scripts_df = scripts_df.sort_values(by='total', ascending=False)

        # Initialize the Dash app
        app = dash.Dash(__name__, title='Scraper Log Analysis')

        # Define the layout
        app.layout = html.Div([
            html.H1('Scraper Log Analysis Dashboard', style={'textAlign': 'center'}),

            html.Div([
                html.Div([
                    html.H3('Overall Statistics'),
                    html.Div([
                        html.P(f"Total articles: {total_articles}"),
                        html.P(f"Relevant articles: {data['total_relevant']}"),
                        html.P(f"Irrelevant articles: {data['total_irrelevant']}"),
                        html.P(f"Relevance ratio: {relevance_ratio:.2f}%"),
                        html.P(f"Total scripts: {len(data['scripts'])}"),
                        html.P(f"Completed: {status_counts['completed']}"),
                        html.P(f"Failed: {status_counts['failed']}"),
                        html.P(f"No data: {status_counts['no_data']}"),
                    ], style={'marginBottom': '20px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

                html.Div([
                    html.H3('Relevant vs Irrelevant Articles'),
                    dcc.Graph(
                        id='pie-chart',
                        figure=px.pie(
                            names=['Relevant', 'Irrelevant'],
                            values=[data['total_relevant'], data['total_irrelevant']],
                            color_discrete_sequence=['#4CAF50', '#F44336'],
                            hole=0.4
                        ).update_layout(
                            margin=dict(t=30, b=10, l=10, r=10),
                            height=300
                        )
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

                html.Div([
                    html.H3('Script Execution Status'),
                    dcc.Graph(
                        id='status-bar',
                        figure=px.bar(
                            x=['Completed', 'Failed', 'No Data'],
                            y=[status_counts['completed'], status_counts['failed'], status_counts['no_data']],
                            color_discrete_sequence=['#4CAF50', '#F44336', '#FFC107']
                        ).update_layout(
                            xaxis_title="Status",
                            yaxis_title="Count",
                            margin=dict(t=30, b=10, l=10, r=10),
                            height=300
                        )
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
            ]),

            html.Div([
                html.H3('Script Performance Analysis'),
                html.Div([
                    html.Label('Sort by:'),
                    dcc.Dropdown(
                        id='sort-by',
                        options=[
                            {'label': 'Total Articles', 'value': 'total'},
                            {'label': 'Relevant Articles', 'value': 'relevant'},
                            {'label': 'Irrelevant Articles', 'value': 'irrelevant'},
                            {'label': 'Relevance Ratio', 'value': 'relevance_ratio'}
                        ],
                        value='total',
                        style={'width': '200px'}
                    ),
                    html.Label('Display top:'),
                    dcc.Slider(
                        id='display-count',
                        min=5,
                        max=min(50, len(scripts_df)),
                        step=5,
                        value=20,
                        marks={i: str(i) for i in range(5, min(50, len(scripts_df))+1, 5)}
                    ),
                ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px'}),

                dcc.Graph(id='script-bar-chart'),

                html.H3('Relevance Ratio by Script'),
                dcc.Graph(id='relevance-bar-chart'),

                html.H3('Script Details'),
                dcc.Dropdown(
                    id='script-selector',
                    options=[{'label': script, 'value': script} for script in scripts_df['script']],
                    value=scripts_df['script'].iloc[0] if not scripts_df.empty else None,
                    style={'width': '100%'}
                ),
                html.Div(id='script-details')
            ], style={'width': '95%', 'margin': 'auto', 'padding': '20px'}),
        ])

        # Define callbacks
        @app.callback(
            [Output('script-bar-chart', 'figure'),
             Output('relevance-bar-chart', 'figure')],
            [Input('sort-by', 'value'),
             Input('display-count', 'value')]
        )
        def update_charts(sort_by, display_count):
            # Sort the DataFrame
            sorted_df = scripts_df.sort_values(by=sort_by, ascending=False).head(display_count)

            # Create article count bar chart
            bar_fig = px.bar(
                sorted_df,
                x='script',
                y=['relevant', 'irrelevant'],
                barmode='group',
                labels={'value': 'Article Count', 'variable': 'Type'},
                color_discrete_sequence=['#4CAF50', '#F44336']
            ).update_layout(
                xaxis_title="Script",
                yaxis_title="Article Count",
                xaxis={'categoryorder': 'total descending'},
                legend_title="Article Type",
                height=500
            )

            # Create relevance ratio bar chart
            relevance_fig = px.bar(
                sorted_df,
                x='script',
                y='relevance_ratio',
                color='relevance_ratio',
                color_continuous_scale=px.colors.sequential.Viridis
            ).update_layout(
                xaxis_title="Script",
                yaxis_title="Relevance Ratio (%)",
                xaxis={'categoryorder': 'total descending'},
                height=500,
                coloraxis_showscale=False
            )

            # Update x-axis layout for better readability
            for fig in [bar_fig, relevance_fig]:
                fig.update_xaxes(tickangle=45, tickfont=dict(size=10))

            return bar_fig, relevance_fig

        @app.callback(
            Output('script-details', 'children'),
            [Input('script-selector', 'value')]
        )
        def display_script_details(script_name):
            if not script_name:
                return html.Div("No script selected")

            script_info = data['scripts'][script_name]

            return html.Div([
                html.H4(f"Details for {script_name}"),
                html.Table([
                    html.Tr([html.Td("Status"), html.Td(script_info['status'])]),
                    html.Tr([html.Td("Total Articles"), html.Td(script_info['total'])]),
                    html.Tr([html.Td("Relevant Articles"), html.Td(script_info['relevant'])]),
                    html.Tr([html.Td("Irrelevant Articles"), html.Td(script_info['irrelevant'])]),
                    html.Tr([html.Td("Relevance Ratio"), html.Td(f"{(script_info['relevant'] / script_info['total'] * 100) if script_info['total'] > 0 else 0:.2f}%")]),
                    html.Tr([html.Td("Vector Errors"), html.Td(script_info.get('vector_errors', 0))]),
                    html.Tr([html.Td("Error Message"), html.Td(script_info.get('error', 'None'))]),
                ], style={'width': '100%', 'border': '1px solid black', 'borderCollapse': 'collapse'})
            ])

        return app

    return Input, Output, create_dashboard, dash, dcc, html, pd, px


@app.cell
def _(create_dashboard, parse_log_data):
    import os

    def main(log_file_path):
        try:
            # Read log file
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_text = f.read()

            # Parse log data
            data = parse_log_data(log_text)

            # Handle empty data safely
            total_articles = data['total_relevant'] + data['total_irrelevant']

            # Print summary
            print(f"Total relevant articles: {data['total_relevant']}")
            print(f"Total irrelevant articles: {data['total_irrelevant']}")

            # Avoid division by zero
            if total_articles > 0:
                print(f"Relevance ratio: {data['total_relevant'] / total_articles * 100:.2f}%")
            else:
                print("Relevance ratio: N/A (no articles processed)")

            print(f"Total scripts processed: {len(data['scripts'])}")

            # Count script statuses
            status_counts = {'completed': 0, 'failed': 0, 'no_data': 0}
            for script in data['scripts']:
                status = data['scripts'][script]['status']
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts['failed'] += 1

            print(f"Scripts completed successfully: {status_counts['completed']}")
            print(f"Scripts failed: {status_counts['failed']}")
            print(f"Scripts with no data: {status_counts['no_data']}")

            # Create dashboard
            app = create_dashboard(data)

            # Create output directory if it doesn't exist
            os.makedirs('output', exist_ok=True)

            # Tell user how to view the dashboard
            print("\nStarting Dash dashboard. Open http://127.0.0.1:8050/ in your browser to view the analysis.")
            app.run(debug=True)

        except FileNotFoundError:
            print(f"Error: Log file '{log_file_path}' not found.")
        except Exception as e:
            print(f"Error analyzing log: {str(e)}")

    return main, os


@app.cell
def _(main, mo):
    import sys

    # Default log file path
    # Handle both local and remote paths for cache
    log_file_path = str(mo.notebook_location() / "public" / "scraper_log.txt")

    # Check for command line arguments
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]

    main(log_file_path)
    return log_file_path, sys


if __name__ == "__main__":
    app.run()
