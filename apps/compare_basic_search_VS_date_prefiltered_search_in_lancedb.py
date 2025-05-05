import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import statistics
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    return go, make_subplots, np, statistics


@app.cell
def _(mo):
    mo.md(
        r"""
    # Compare basic search VS date prefiltered search in lancedb

    I tested two cases, each with two scenarios

     - **basic** - no special filters just fetches url of a certain domain
        - **limited** - fetches limited amout (50 for this dataset) 
        - **unlimited** - all urls
     - **prefiltered by date** - fetches  urls of a certain domain but prefilteres it by date first
        - **limited** - fetches limited amout (50 for this dataset) 
        - **unlimited** - all urls  (kinda useless situation)

    i removed the outliers as much as i can. especially so for the **prefiltered by date**. i did it in a way to give it an advantage over **basic**

    i tested around 22 times for all of these 4 situations

    i was also intrested if size matters since we are fetching through the network. (usually bigger is slower)

    ##Results: 
     - basic is better.
     - see the graphs below for why
    ## Steps:
    ### Dataset
    """
    )
    return


@app.cell
def _():

    dataset_unlimited_basic = [
        7.02, 6.86, 6.81, 7.17, 6.65, 6.73, 7.01, 6.78, 7.60, 6.72, 7.20, 
        7.04, 6.73, 6.65, 6.81, 7.38, 7.43, 7.41, 7.22, 7.05, 6.87, 
        7.09, 7.04, 6.72, 7.10
    ]

    dataset_unlimited_dated = [
        7.26, 6.79, 6.26, 7.02, 6.60, 7.53, 7.10, 6.61, 6.99, 8.15, 
        9.73, 7.03, 8.58, 6.47, 6.78, 8.69, 7.52, 6.99, 6.50, 6.98, 
        6.53, 6.44
    ]

    dataset_limited_basic = [
        5.28, 5.82, 7.85, 4.82, 4.97, 8.52, 5.56, 8.71, 4.70, 7.52,
        5.03, 4.74, 7.86, 4.53, 5.15, 6.16, 5.01, 4.99, 6.67, 8.12,
        5.18, 5.24, 5.51, 5.09
    ]

    dataset_limited_dated = [
        6.83, 7.97, 6.24, 6.72, 8.30, 9.66, 11.10, 7.07, 7.65, 10.12,
        8.44, 8.76, 6.22, 7.84, 7.27, 6.04, 6.80, 7.54, 6.14, 6.31,
        9.90, 7.93, 7.44, 8.24
    ]

    return (
        dataset_limited_basic,
        dataset_limited_dated,
        dataset_unlimited_basic,
        dataset_unlimited_dated,
    )


@app.cell
def _(
    dataset_limited_basic,
    dataset_limited_dated,
    dataset_unlimited_basic,
    dataset_unlimited_dated,
    np,
    statistics,
):

    # Dataset metadata
    dataset_names = [
        "Unlimited Basic (11.2 kB)",
        "Unlimited Dated (11.9 kB)",
        "Limited Basic (5.0 kB)",
        "Limited Dated (7.7 kB)"
    ]

    dataset_sizes = [11.2, 11.9, 5.0, 7.7]  # in kB
    datasets = [dataset_unlimited_basic, dataset_unlimited_dated, dataset_limited_basic, dataset_limited_dated]

    # Analysis function
    def analyze_dataset(data):
        return {
            "count": len(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "std_dev": statistics.stdev(data)
        }

    # Calculate statistics for all datasets
    stats = [analyze_dataset(dataset) for dataset in datasets]

    # Create a DataFrame-like structure for easier analysis
    results = []
    for i, (name, size, dataset, stat) in enumerate(zip(dataset_names, dataset_sizes, datasets, stats)):
        results.append({
            "id": i,
            "name": name,
            "size_kb": size,
            "count": stat["count"],
            "mean": stat["mean"],
            "median": stat["median"],
            "min": stat["min"],
            "max": stat["max"],
            "range": stat["range"],
            "std_dev": stat["std_dev"]
        })

    return dataset_names, dataset_sizes, datasets, results, stats


@app.cell
def _(dataset_names, dataset_sizes, datasets, go, make_subplots, np, stats):


    # 1. Box plot comparing response times across datasets
    fig1 = go.Figure()
    for dist_dataset, dist_name in zip(datasets, dataset_names):
        fig1.add_trace(go.Box(
            y=dist_dataset,
            name=dist_name,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=4),
        ))
    fig1.update_layout(
        title='Response Time Distribution by Dataset',
        yaxis_title='Response Time (seconds)',
        showlegend=True
    )

    # 2. Bar chart for average response times with error bars
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=dataset_names,
        y=[s["mean"] for s in stats],
        error_y=dict(
            type='data',
            array=[s["std_dev"] for s in stats],
            visible=True
        ),
        text=[f"{s['mean']:.2f}s" for s in stats],
        textposition='auto',
    ))
    fig2.update_layout(
        title='Average Response Time by Dataset',
        xaxis_title='Dataset',
        yaxis_title='Average Response Time (seconds)'
    )

    # 3. Scatter plot for correlation between response size and average time
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=dataset_sizes,
        y=[s["mean"] for s in stats],
        mode='markers+text',
        text=dataset_names,
        textposition="top center",
        marker=dict(size=12)
    ))

    # Add trendline
    slope, intercept = np.polyfit(dataset_sizes, [s["mean"] for s in stats], 1)
    correlation = np.corrcoef(dataset_sizes, [s["mean"] for s in stats])[0, 1]
    x_range = np.linspace(min(dataset_sizes) - 1, max(dataset_sizes) + 1, 100)
    fig3.add_trace(go.Scatter(
        x=x_range,
        y=slope * x_range + intercept,
        mode='lines',
        name=f'Trend: y={slope:.4f}x+{intercept:.4f}, r={correlation:.4f}'
    ))
    fig3.update_layout(
        title='Correlation: Response Size vs Average Time',
        xaxis_title='Response Size (kB)',
        yaxis_title='Average Time (seconds)'
    )

    # 4. Violin plot to show distribution details
    fig4 = go.Figure()
    for dist_dataset, dist_name in zip(datasets, dataset_names):
        fig4.add_trace(go.Violin(
            y=dist_dataset,
            name=dist_name,
            box_visible=True,
            meanline_visible=True,
            points='all'
        ))
    fig4.update_layout(
        title='Response Time Distribution (Violin Plot)',
        yaxis_title='Response Time (seconds)'
    )

    # 5. Histogram comparing the frequency distributions
    fig5 = make_subplots(rows=2, cols=2, subplot_titles=dataset_names)

    for dist, (dist_dataset, dist_name) in enumerate(zip(datasets, dataset_names)):
        row = dist // 2 + 1
        col = dist % 2 + 1
        fig5.add_trace(
            go.Histogram(
                x=dist_dataset,
                nbinsx=10,
                name=dist_name
            ),
            row=row, col=col
        )

    fig5.update_layout(
        title_text='Response Time Frequency Distributions',
        showlegend=False
    )

    print("---")
    return correlation, fig1, fig2, fig3, fig4, fig5, intercept, slope


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Stats summary
    ### Create visualizations
    """
    )
    return


@app.cell
def _(correlation, intercept, mo, results, slope):
    summary_text = "# Stats Summary\n\n"
    for result in results:
        summary_text += f"## {result['name']} (Size: {result['size_kb']} kB)\n"
        summary_text += f"- **Mean**: {result['mean']:.2f} seconds\n"
        summary_text += f"- **Median**: {result['median']:.2f} seconds\n"
        summary_text += f"- **Range**: {result['min']:.2f} - {result['max']:.2f} seconds\n"
        summary_text += f"- **Std Dev**: {result['std_dev']:.3f} seconds\n\n"

    # Add correlation information
    summary_text += f"### Correlation Analysis\n"
    summary_text += f"- **Correlation coefficient**: {correlation:.3f}\n"
    summary_text += f"- **Regression equation**: Time = {slope:.4f} × Size + {intercept:.4f}\n"

    # Display the formatted markdown
    mo.md(summary_text)
    return


@app.cell
def _(mo):
    mo.md(r"""# Display all plots (its interactive)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(fig1):
    fig1
    return


@app.cell
def _(fig2):
    fig2
    return


@app.cell
def _(fig3):
    fig3
    return


@app.cell
def _(fig4):
    fig4
    return


@app.cell
def _(fig5):
    fig5
    return


if __name__ == "__main__":
    app.run()
