

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import requests

    def read_log_file(log_file_path):
        """Read log file from either local path or URL"""
        # Check if it's a URL (starts with http:// or https://)
        if log_file_path.startswith(('http://', 'https://')):
            try:
                response = requests.get(log_file_path)
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                return response.content
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to download log file: {e}")
        else:
            # Assume it's a local file path
            try:
                with open(log_file_path, 'rb') as f:
                    return f.read()
            except FileNotFoundError:
                raise Exception(f"Log file not found: {log_file_path}")
            except Exception as e:
                raise Exception(f"Failed to read log file: {e}")
    return (read_log_file,)


@app.cell
def _():
    import re
    from datetime import datetime


    def parse_log_data(log_text):
        # First, check if log_text is bytes and convert to string if needed
        if isinstance(log_text, bytes):
            log_text = log_text.decode('utf-8')  # or other appropriate encoding
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
    return (parse_log_data,)


@app.cell
def _():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def generate_plotly_visualizations(data):
        # Create a list of figures to return
        figures = []

        # 1. Overall Relevance Pie Chart
        total_relevant = data['total_relevant']
        total_irrelevant = data['total_irrelevant']
        total = total_relevant + total_irrelevant

        if total > 0:
            labels = ['Relevant', 'Not Relevant']
            values = [total_relevant, total_irrelevant]

            fig1 = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=['#66c2a5', '#fc8d62']
            )])
            fig1.update_layout(
                title_text='Overall Article Relevance',
                annotations=[dict(text=f'{total} Articles', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            figures.append(fig1)

        # 2. Script Status Summary
        status_counts = {'completed': 0, 'failed': 0, 'no_data': 0, 'started': 0}
        for script in data['scripts']:
            status = data['scripts'][script]['status']
            status_counts[status] += 1

        fig2 = go.Figure(data=[go.Bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
            marker_color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        )])
        fig2.update_layout(
            title_text='Script Status Summary',
            xaxis_title='Status',
            yaxis_title='Count'
        )
        figures.append(fig2)

        # 3. Per-Script Relevance
        if data['scripts']:
            # Convert to DataFrame for easier plotting
            script_df = []
            for script, stats in data['scripts'].items():
                script_df.append({
                    'script': script,
                    'relevant': stats.get('relevant', 0),
                    'irrelevant': stats.get('irrelevant', 0),
                    'total': stats.get('total', 0),
                    'status': stats.get('status', 'unknown')
                })

            script_df = pd.DataFrame(script_df)
            if not script_df.empty:
                # Sort by total articles
                script_df = script_df.sort_values('total', ascending=False)

                # Keep top 10 scripts for readability
                if len(script_df) > 10:
                    script_df = script_df.head(10)
                    title = 'Top 10 Scripts by Article Count'
                else:
                    title = 'Scripts by Article Count'

                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=script_df['script'],
                    y=script_df['relevant'],
                    name='Relevant',
                    marker_color='#66c2a5'
                ))
                fig3.add_trace(go.Bar(
                    x=script_df['script'],
                    y=script_df['irrelevant'],
                    name='Not Relevant',
                    marker_color='#fc8d62'
                ))

                fig3.update_layout(
                    title_text=title,
                    xaxis_title='Script',
                    yaxis_title='Article Count',
                    barmode='stack',
                    xaxis={'tickangle': 45}
                )
                figures.append(fig3)

        # 4. Relevance Rate by Script
        if data['scripts']:
            relevance_rates = []
            script_names = []
            for script, stats in data['scripts'].items():
                total = stats.get('total', 0)
                if total > 0:
                    relevance_rate = (stats.get('relevant', 0) / total) * 100
                    relevance_rates.append(relevance_rate)
                    script_names.append(script)

            if relevance_rates:
                # Create DataFrame and sort
                rate_df = pd.DataFrame({
                    'script': script_names,
                    'relevance_rate': relevance_rates
                })
                rate_df = rate_df.sort_values('relevance_rate', ascending=False)

                # Keep top 10 for readability
                if len(rate_df) > 10:
                    rate_df = rate_df.head(10)
                    title = 'Top 10 Scripts by Relevance Rate'
                else:
                    title = 'Scripts by Relevance Rate'

                fig4 = go.Figure(data=[
                    go.Bar(
                        x=rate_df['script'],
                        y=rate_df['relevance_rate'],
                        marker_color='#8da0cb'
                    )
                ])
                fig4.update_layout(
                    title_text=title,
                    xaxis_title='Script',
                    yaxis_title='Relevance Rate (%)',
                    xaxis={'tickangle': 45}
                )
                figures.append(fig4)

        return figures
    return (generate_plotly_visualizations,)


@app.cell
def _(generate_plotly_visualizations, parse_log_data, read_log_file):
    import os

    def main(log_file_path):
        try:
            # Read log file
            log_text = read_log_file(log_file_path)


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
                print(script)
                status = data['scripts'][script]['status']
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts['failed'] += 1

            print(f"Scripts completed successfully: {status_counts['completed']}")
            print(f"Scripts failed: {status_counts['failed']}")
            print(f"Scripts with no data: {status_counts['no_data']}")

            # Generate and display plotly figures
            figures = generate_plotly_visualizations(data)

            # Display the figures
            for fig in figures:
                fig.show()

            return data  # Return the data for further analysis if needed

        except Exception as e:
            print(f"Error analyzing log: {str(e)}")
            import traceback
            traceback.print_exc()

    return (main,)


@app.cell
def _(main, mo):
    # Default log file path
    # Handle both local and remote paths for cache
    log_file_path = str(mo.notebook_location() / "public" / "scraper_log.txt")

    main(log_file_path)
    return


if __name__ == "__main__":
    app.run()
