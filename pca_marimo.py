import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Analysis Demo - PCA Notebook (Marimo)
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import pandas as pd

    from tasks import SingleTrialTask
    from models.rnn import RNN
    from analysis import (
        generate_data,
        do_pca,
        compute_psychometric_curves,
    )

    plt.rcParams["figure.dpi"] = 100
    return (
        RNN,
        SingleTrialTask,
        compute_psychometric_curves,
        do_pca,
        generate_data,
        np,
        pd,
        px,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup: Load Model and Generate Data
    """)
    return


@app.cell
def _(mo):
    # Configuration controls
    checkpoint_selector = mo.ui.text(
        value="logs/single_trial/checkpoints/2000.pt", label="Checkpoint path:"
    )

    num_trials_slider = mo.ui.slider(
        start=20, stop=200, step=10, value=80, label="Number of trials to generate:"
    )

    trials_per_seq_slider = mo.ui.slider(
        start=10, stop=100, step=10, value=40, label="Trials per sequence:"
    )

    mo.vstack([checkpoint_selector, num_trials_slider, trials_per_seq_slider])
    return checkpoint_selector, num_trials_slider, trials_per_seq_slider


@app.cell
def _(
    RNN,
    SingleTrialTask,
    checkpoint_selector,
    generate_data,
    mo,
    np,
    num_trials_slider,
    torch,
    trials_per_seq_slider,
):
    checkpoint_path = checkpoint_selector.value
    num_trials = num_trials_slider.value

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        task = SingleTrialTask(
            w_m=0.05,
        )

        model = RNN(
            hidden_size=128,
            noise_std=0.0,  # No noise during evaluation
        )

        # Load trained weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Generate data
        data_dict = generate_data(task, model, num_trials=num_trials)
        rules = np.array(
            [
                data_dict["batch"][i]["metadata"]["rule"][0]
                for i in range(len(data_dict["batch"]))
            ]
        )

        setup_status = mo.md(f"""
        **Model loaded successfully!**
        - Checkpoint step: {checkpoint["step"]}
        - Hidden units: {model.hidden_size}
        - Trials per sequence: {task.trials_per_sequence}
        - Generated trials: {num_trials}
        - Rule 1 count: {(rules == 1).sum()}, Rule 2 count: {(rules == -1).sum()}
        """)

    except Exception as e:
        setup_status = mo.md(f"**Error loading checkpoint:** {str(e)}")
        data_dict = None
        task = None
        model = None
        rules = None

    setup_status
    return data_dict, model, task


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Random Trial Visualization
    """)
    return


@app.cell
def _(data_dict, mo, np):
    # Trial selector
    if data_dict is not None:
        trial_selector = mo.ui.slider(
            start=0,
            stop=len(data_dict["batch"]) - 1,
            step=1,
            value=np.random.randint(0, len(data_dict["batch"])),
            label="Trial index:",
            show_value=True,
        )

        randomize_button = mo.ui.button(label="Random Trial")
    else:
        trial_selector = None
        randomize_button = None

    mo.hstack([trial_selector, randomize_button]) if trial_selector else mo.md(
        "*Load data first*"
    )
    return randomize_button, trial_selector


@app.cell
def _(data_dict, mo, np, randomize_button, trial_selector):
    # Handle randomization
    if randomize_button is not None and randomize_button.value:
        trial_idx = np.random.randint(0, len(data_dict["batch"]))
    elif trial_selector is not None:
        trial_idx = trial_selector.value
    else:
        trial_idx = 0

    mo.md(f"**Showing trial:** {trial_idx}")
    return (trial_idx,)


@app.cell
def _(data_dict, mo, task, trial_idx):
    if data_dict is not None and task is not None:
        _batch = data_dict["batch"]
        _outputs = data_dict["outputs"]

        trial_outputs = _outputs[trial_idx].numpy()
        trial_inputs = _batch[trial_idx]["inputs"][0].numpy()
        trial_targets = _batch[trial_idx]["targets"][0].numpy()
        trial_eval_mask = _batch[trial_idx]["eval_mask"][0].numpy()
        trial_loss_mask = _batch[trial_idx]["loss_mask"][0].numpy()

        # Create figure using task method
        trial_plot = task.create_trial_figure(
            inputs=trial_inputs,
            outputs=trial_outputs,
            targets=trial_targets,
            eval_mask=trial_eval_mask,
            trial_idx=trial_idx,
            batch=_batch,
            batch_idx=0,
            loss_mask=trial_loss_mask,
        )
    else:
        trial_plot = mo.md("*Load data first*")

    trial_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## PCA Visualization
    """)
    return


@app.cell
def _(mo):
    # PCA controls
    n_components_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label="Number of PCA components:",
        show_value=True,
    )

    plot_3d_checkbox = mo.ui.checkbox(value=True, label="3D plot")

    animate_checkbox = mo.ui.checkbox(value=False, label="Animate over time")

    num_trials_pca_slider = mo.ui.slider(
        start=10,
        stop=80,
        step=10,
        value=40,
        label="Number of trials to visualize:",
        show_value=True,
    )

    color_by_dropdown = mo.ui.dropdown(
        options=[
            "rule",
            "decision",
            "stim_direction",
            "instructed",
            "switch",
            "reward",
            "t_m",
            "t_s",
            "trial",
        ],
        value="rule",
        label="Color by:",
    )

    mo.hstack(
        [
            mo.vstack([n_components_slider, num_trials_pca_slider]),
            mo.vstack([plot_3d_checkbox, animate_checkbox, color_by_dropdown]),
        ]
    )
    return (
        animate_checkbox,
        color_by_dropdown,
        n_components_slider,
        num_trials_pca_slider,
        plot_3d_checkbox,
    )


@app.cell(hide_code=True)
def _(
    animate_checkbox,
    color_by_dropdown,
    data_dict,
    do_pca,
    mo,
    model,
    n_components_slider,
    num_trials_pca_slider,
    pd,
    plot_3d_checkbox,
    px,
    task,
):
    if data_dict is not None and task is not None:
        # Run PCA
        result_full = do_pca(
            data_dict, task, model=model, n_components=n_components_slider.value
        )

        # Prepare data for plotly
        pca_data = result_full["pca_data"]
        axis_labels = result_full["axis_labels"]
        metadata = result_full["metadata"]
        lengths = result_full["lengths"]

        # Limit to num_trials
        num_show = min(num_trials_pca_slider.value, len(pca_data))

        # Define color schemes (matching matplotlib version)
        color_by = color_by_dropdown.value
        _continuous_schemes = ["t_m", "t_s"]
        _is_continuous = color_by in _continuous_schemes

        # Build dataframe for plotly
        # For animation, we need cumulative data (each frame shows trajectory up to that point)
        # For static plots, just show all points once
        plot_rows = []

        if animate_checkbox.value:
            # Cumulative data: for each animation frame, include all timesteps up to that frame
            max_len = max(int(lengths[i]) for i in range(num_show))
            for _trial_idx in range(num_show):
                trial_len = int(lengths[_trial_idx])

                # Get color value for this trial
                if color_by == "trial":
                    color_val = f"Trial {_trial_idx + 1}"
                elif color_by in metadata:
                    if color_by == "rule":
                        color_val = (
                            "Rule 1" if metadata["rule"][_trial_idx] == 1 else "Rule 2"
                        )
                    elif color_by == "decision":
                        color_val = (
                            "Right"
                            if metadata.get("decision", [1])[_trial_idx] == 1
                            else "Left"
                        )
                    elif color_by == "stim_direction":
                        color_val = (
                            "Right"
                            if metadata["stim_direction"][_trial_idx] == 1
                            else "Left"
                        )
                    elif color_by == "instructed":
                        val = metadata["instructed"][_trial_idx]
                        color_val = (
                            "Instructed" if (val or val == 1) else "Uninstructed"
                        )
                    elif color_by == "switch":
                        val = metadata.get("switch", [0])[_trial_idx]
                        color_val = "Switch" if (val or val == 1) else "No Switch"
                    elif color_by == "reward":
                        val = metadata.get("reward", [0])[_trial_idx]
                        color_val = "Rewarded" if (val == 1 or val) else "Not Rewarded"
                    elif _is_continuous:
                        color_val = float(metadata[color_by][_trial_idx])
                    else:
                        color_val = f"Trial {_trial_idx + 1}"
                else:
                    color_val = f"Trial {_trial_idx + 1}"

                # For each animation frame, add all points up to that frame
                for frame in range(max_len):
                    for t in range(min(frame + 1, trial_len)):
                        row = {
                            "trial": _trial_idx,
                            "timestep_actual": t,
                            "animation_frame": frame,
                            axis_labels[0]: pca_data[_trial_idx, t, 0],
                            axis_labels[1]: pca_data[_trial_idx, t, 1],
                            "color": color_val,
                        }
                        if pca_data.shape[2] >= 3:
                            row[axis_labels[2]] = pca_data[_trial_idx, t, 2]
                        plot_rows.append(row)
        else:
            # Static data: just show all points
            for _trial_idx in range(num_show):
                trial_len = int(lengths[_trial_idx])

                # Get color value for this trial
                if color_by == "trial":
                    color_val = f"Trial {_trial_idx + 1}"
                elif color_by in metadata:
                    if color_by == "rule":
                        color_val = (
                            "Rule 1" if metadata["rule"][_trial_idx] == 1 else "Rule 2"
                        )
                    elif color_by == "decision":
                        color_val = (
                            "Right"
                            if metadata.get("decision", [1])[_trial_idx] == 1
                            else "Left"
                        )
                    elif color_by == "stim_direction":
                        color_val = (
                            "Right"
                            if metadata["stim_direction"][_trial_idx] == 1
                            else "Left"
                        )
                    elif color_by == "instructed":
                        val = metadata["instructed"][_trial_idx]
                        color_val = (
                            "Instructed" if (val or val == 1) else "Uninstructed"
                        )
                    elif color_by == "switch":
                        val = metadata.get("switch", [0])[_trial_idx]
                        color_val = "Switch" if (val or val == 1) else "No Switch"
                    elif color_by == "reward":
                        val = metadata.get("reward", [0])[_trial_idx]
                        color_val = "Rewarded" if (val == 1 or val) else "Not Rewarded"
                    elif _is_continuous:
                        color_val = float(metadata[color_by][_trial_idx])
                    else:
                        color_val = f"Trial {_trial_idx + 1}"
                else:
                    color_val = f"Trial {_trial_idx + 1}"

                for t in range(trial_len):
                    row = {
                        "trial": _trial_idx,
                        "timestep": t,
                        axis_labels[0]: pca_data[_trial_idx, t, 0],
                        axis_labels[1]: pca_data[_trial_idx, t, 1],
                        "color": color_val,
                    }
                    if pca_data.shape[2] >= 3:
                        row[axis_labels[2]] = pca_data[_trial_idx, t, 2]
                    plot_rows.append(row)

        df = pd.DataFrame(plot_rows)

        # Create plotly figure
        # For animation, use lines with cumulative data; for static, use lines
        if animate_checkbox.value:
            # Animated line plots (cumulative trajectory builds up)
            if plot_3d_checkbox.value and pca_data.shape[2] >= 3:
                _fig = px.line_3d(
                    df,
                    x=axis_labels[0],
                    y=axis_labels[1],
                    z=axis_labels[2],
                    color="color",
                    line_group="trial",
                    hover_data=["trial", "timestep_actual"],
                    title="PCA Trajectories (3D) - Animated",
                    animation_frame="animation_frame",
                    range_x=[df[axis_labels[0]].min(), df[axis_labels[0]].max()],
                    range_y=[df[axis_labels[1]].min(), df[axis_labels[1]].max()],
                    range_z=[df[axis_labels[2]].min(), df[axis_labels[2]].max()],
                )
                _fig.update_traces(opacity=0.6, line=dict(width=3))
            else:
                _fig = px.line(
                    df,
                    x=axis_labels[0],
                    y=axis_labels[1],
                    color="color",
                    line_group="trial",
                    hover_data=["trial", "timestep_actual"],
                    title="PCA Trajectories (2D) - Animated",
                    animation_frame="animation_frame",
                    range_x=[df[axis_labels[0]].min(), df[axis_labels[0]].max()],
                    range_y=[df[axis_labels[1]].min(), df[axis_labels[1]].max()],
                )
                _fig.update_traces(opacity=0.6, line=dict(width=3))
        else:
            # Static line plots (no animation)
            if plot_3d_checkbox.value and pca_data.shape[2] >= 3:
                _fig = px.line_3d(
                    df,
                    x=axis_labels[0],
                    y=axis_labels[1],
                    z=axis_labels[2],
                    color="color",
                    line_group="trial",
                    hover_data=["trial", "timestep"],
                    title="PCA Trajectories (3D)",
                )
                _fig.update_traces(opacity=0.6)
            else:
                _fig = px.line(
                    df,
                    x=axis_labels[0],
                    y=axis_labels[1],
                    color="color",
                    line_group="trial",
                    hover_data=["trial", "timestep"],
                    title="PCA Trajectories (2D)",
                )
                _fig.update_traces(opacity=0.6)

        _fig.update_layout(height=600)

        pca_info = mo.md(f"""
        **PCA Results:**
        - Data shape: {result_full["pca_data"].shape}
        - Axis labels: {result_full["axis_labels"]}
        """)

        pca_plot = mo.ui.plotly(_fig)
    else:
        result_full = None
        pca_plot = mo.md("*Load data first*")
        pca_info = mo.md("")

    mo.vstack([pca_info, pca_plot])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cross-Period Variance

    How well do PCs from one period explain variance in other periods?
    """)
    return


@app.cell
def _(mo):
    # Cross-period variance controls
    n_components_cross_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label="Number of components for cross-period analysis:",
        show_value=True,
    )

    n_components_cross_slider
    return (n_components_cross_slider,)


@app.cell
def _(data_dict, do_pca, mo, n_components_cross_slider, np, px, task):
    if data_dict is not None and task is not None:
        # Determine period names
        _batch_cross = data_dict["batch"]
        _hidden_states = data_dict["hidden_states"]

        period_names = ["rule_report", "timing", "decision"]
        # Check if this is a sequence task with ITI
        if len(_batch_cross) > 1 or (
            len(_batch_cross) == 1 and "iti_start" in _batch_cross[0]["metadata"]
        ):
            period_names.append("iti")

        # Map 'iti' to 'post_iti' for internal use
        internal_period_names = [p if p != "iti" else "post_iti" for p in period_names]

        # Create display labels
        display_labels = []
        for p in period_names:
            if p == "iti":
                display_labels.append("ITI")
            else:
                display_labels.append(p.replace("_", " ").title())

        n_periods = len(period_names)
        n_comp = n_components_cross_slider.value

        # Get PCA results for each period
        period_pcs = {}
        period_data = {}

        for internal_name in internal_period_names:
            result = do_pca(data_dict, task, periods=internal_name, n_components=n_comp)
            pcs = result["pcs"]  # [H, n_comp]

            # For simplicity, recompute: extract period boundaries and get raw data
            from analysis.pca import _extract_period_boundaries

            _num_trials = _hidden_states.shape[0]
            period_info_dict = _extract_period_boundaries(
                _batch_cross, task, _num_trials
            )
            period_boundaries = period_info_dict[internal_name]

            hidden_np = _hidden_states.cpu().numpy()

            # Flatten period data
            flat_data_list = []
            for i, info in enumerate(period_boundaries):
                start, end = info["start"], info["end"]
                period_len = end - start
                if period_len == 0:
                    continue
                _trial_idx_cross = info.get("trial_idx", i)
                flat_data_list.append(hidden_np[_trial_idx_cross, :, start:end].T)

            hidden_flat = np.vstack(flat_data_list)

            period_pcs[internal_name] = pcs
            period_data[internal_name] = hidden_flat

        # Compute variance matrix
        variance_matrix = np.zeros((n_periods, n_periods))

        for i, fit_period in enumerate(internal_period_names):
            fit_pcs = period_pcs[fit_period]

            for j, test_period in enumerate(internal_period_names):
                test_data = period_data[test_period]
                test_centered = test_data - test_data.mean(axis=0)
                test_projected = test_centered @ fit_pcs

                projected_variance = np.sum(np.var(test_projected, axis=0))
                total_variance = np.sum(np.var(test_centered, axis=0))
                variance_explained = projected_variance / total_variance

                variance_matrix[i, j] = variance_explained * 100

        # Create plotly heatmap
        _fig_cross = px.imshow(
            variance_matrix,
            labels=dict(x="Test", y="Fit", color="Variance (%)"),
            x=display_labels,
            y=display_labels,
            color_continuous_scale="viridis",
            zmin=0,
            zmax=100,
            title=f"Cross-Period Variance Explained by Top {n_comp} PCs (%)",
            text_auto=".1f",
        )

        _fig_cross.update_layout(height=500)
        cross_period_plot = _fig_cross
    else:
        cross_period_plot = mo.md("*Load data first*")

    cross_period_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Psychometric Function

    P(Anti saccade) vs interval duration for each rule.
    """)
    return


@app.cell
def _(mo):
    # Psychometric function controls
    num_trials_per_interval_slider = mo.ui.slider(
        start=50,
        stop=500,
        step=50,
        value=100,
        label="Trials per interval:",
        show_value=True,
    )

    num_trials_per_interval_slider
    return (num_trials_per_interval_slider,)


@app.cell
def _(
    compute_psychometric_curves,
    mo,
    model,
    num_trials_per_interval_slider,
    pd,
    px,
    task,
):
    if model is not None and task is not None:
        # Compute psychometric curves
        psycho_results = compute_psychometric_curves(
            task,
            model,
            num_trials_per_interval=num_trials_per_interval_slider.value,
            rules=[1, -1],
        )

        # Build dataframe for plotly
        psycho_rows = []
        for rule in [1, -1]:
            rule_label = "Rule 1" if rule == 1 else "Rule 2"
            anti_probs = psycho_results["anti_probs"][rule]
            for interval, prob in zip(psycho_results["intervals"], anti_probs):
                psycho_rows.append(
                    {
                        "Interval (ms)": interval,
                        "P(Anti saccade)": prob,
                        "Rule": rule_label,
                    }
                )

        df_psycho = pd.DataFrame(psycho_rows)

        # Create plotly figure
        _fig_psycho = px.line(
            df_psycho,
            x="Interval (ms)",
            y="P(Anti saccade)",
            color="Rule",
            markers=True,
            title="Psychometric Curves",
        )

        # Add threshold line
        _fig_psycho.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        _fig_psycho.add_vline(
            x=task.decision_threshold,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Threshold ({task.decision_threshold}ms)",
            annotation_position="top right",
        )

        _fig_psycho.update_layout(height=400)

        psycho_plot = _fig_psycho
    else:
        psycho_results = None
        psycho_plot = mo.md("*Load data first*")

    psycho_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Variance Explained

    How many principal components are needed to capture neural variance?
    """)
    return


@app.cell
def _(mo):
    # Variance explained controls
    n_components_var_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=20,
        label="Number of components to analyze:",
        show_value=True,
    )

    n_components_var_slider
    return (n_components_var_slider,)


@app.cell
def _(data_dict, do_pca, mo, n_components_var_slider, np, pd, px, task):
    if data_dict is not None and task is not None:
        # Perform PCA with many components
        result_dims = do_pca(
            data_dict, task, periods="all", n_components=n_components_var_slider.value
        )

        # Extract variance explained
        var_explained = result_dims["explained_variance"]
        var_explained_pct = 100 * var_explained / var_explained.sum()
        cumulative_var = np.cumsum(var_explained_pct)

        # Summary statistics
        n_90 = (
            np.argmax(cumulative_var >= 90) + 1
            if np.any(cumulative_var >= 90)
            else len(cumulative_var)
        )
        n_95 = (
            np.argmax(cumulative_var >= 95) + 1
            if np.any(cumulative_var >= 95)
            else len(cumulative_var)
        )

        var_summary = mo.md(f"""
        **Dimensionality Summary:**
        - Components for 90% variance: {n_90}
        - Components for 95% variance: {n_95}
        - Top 5 PCs explain: {cumulative_var[4]:.1f}% of variance
        - Top 10 PCs explain: {cumulative_var[9]:.1f}% of variance
        """)

        # Create bar plot dataframe
        x = np.arange(1, len(var_explained_pct) + 1)
        df_bar = pd.DataFrame({"PC": x, "Variance (%)": var_explained_pct})

        _fig_bar = px.bar(
            df_bar, x="PC", y="Variance (%)", title="Variance Explained by Each PC"
        )
        _fig_bar.update_layout(height=400)

        # Create cumulative plot dataframe
        df_cum = pd.DataFrame(
            {"Number of Components": x, "Cumulative Variance (%)": cumulative_var}
        )

        _fig_cum = px.line(
            df_cum,
            x="Number of Components",
            y="Cumulative Variance (%)",
            title="Cumulative Variance Explained",
            markers=True,
        )
        _fig_cum.add_hline(
            y=90, line_dash="dash", line_color="red", opacity=0.5, annotation_text="90%"
        )
        _fig_cum.add_hline(
            y=95,
            line_dash="dash",
            line_color="orange",
            opacity=0.5,
            annotation_text="95%",
        )
        _fig_cum.update_layout(height=400, yaxis_range=[0, 105])

        var_plot = mo.hstack([_fig_bar, _fig_cum])
    else:
        result_dims = None
        var_summary = mo.md("*Load data first*")
        var_plot = mo.md("")

    mo.vstack([var_summary, var_plot])
    return


if __name__ == "__main__":
    app.run()
