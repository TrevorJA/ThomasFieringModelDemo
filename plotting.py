
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_flow_ranges(Qh, Qs, 
                     timestep = 'daily',
                     units = 'cms', y_scale = 'log',
                     savefig = False, fig_dir = '.',
                     figsize = (7,5), colors = ['black', 'orange'],
                     title_addon = ""):
    """Plots the range of flow for historic and syntehtic streamflows for a specific timestep scale.
    
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_addon (str, optional): Text to be added to the end of the title. Defaults to "".
    """


    # Assert formatting matches expected
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'

    # Handle conditional datetime formatting
    if timestep == 'daily':
        h_grouper = Qh.index.dayofyear
        s_grouper = Qs.index.dayofyear
        x_lab = 'Day of the Year (Jan-Dec)'
    elif timestep == 'monthly':
        h_grouper = Qh.index.month
        s_grouper = Qs.index.month
        x_lab = 'Month of the Year (Jan-Dec)'
    elif timestep == 'weekly':
        h_grouper = pd.Index(Qh.index.isocalendar().week, dtype = int)
        s_grouper = pd.Index(Qs.index.isocalendar().week, dtype = int)
        x_lab = 'Week of the Year (Jan-Dec)'
    else:
        print('Invalid timestep input. Options: "daily", "monthly", "weekly".')
        return

    # Find flow ranges
    s_max = Qs.groupby(s_grouper).max().max(axis=1)
    s_min = Qs.groupby(s_grouper).min().min(axis=1)
    s_median = Qs.groupby(s_grouper).median().median(axis=1)
    h_max = Qh.groupby(h_grouper).max()
    h_min = Qh.groupby(h_grouper).min()
    h_median = Qh.groupby(h_grouper).median()
  
    ## Plotting  
    fig, ax = plt.subplots(figsize = figsize, dpi=150)
    xs = h_max.index
    ax.fill_between(xs, s_min, s_max, color = colors[1], label = 'Synthetic Range', alpha = 0.5)
    ax.plot(xs, s_median, color = colors[1], label = 'Synthetic Median')
    ax.fill_between(xs, h_min, h_max, color = colors[0], label = 'Historic Range', alpha = 0.3)
    ax.plot(xs, h_median, color = colors[0], label = 'Historic Median')
    
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'{timestep.capitalize()} Flow ({units})', fontsize=12)
    ax.set_xlabel(x_lab, fontsize=12)

    ax.legend(ncols = 2, fontsize = 10, bbox_to_anchor = (0, -.5, 1.0, 0.2), loc = 'upper center')    
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title(f'{timestep.capitalize()} Streamflow Ranges\nHistoric & Synthetic Timeseries at One Location\n{title_addon}')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'{fig_dir}/flow_ranges_{timestep}.png', dpi = 150)
    return


def plot_fdc_ranges(Qh, Qs, 
                    units = 'cms', y_scale = 'log',
                    savefig = False, fig_dir = '.',
                    figsize = (5,5), colors = ['black', 'orange'],                   
                    title_addon = ""):
    """Plots the range and aggregate flow duration curves for historic and synthetic streamflows.
    
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_addon (str, optional): Text to be added to the end of the title. Defaults to "".
    """
    
    ## Assertions
    assert(type(Qs) == pd.DataFrame), 'Synthetic streamflow should be type pd.DataFrame.'
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'

    
    # Calculate FDCs for total period and each realization
    nonexceedance = np.linspace(0.0001, 0.9999, 50)
    s_total_fdc = np.quantile(Qs.values.flatten(), nonexceedance)
    h_total_fdc = np.quantile(Qh.values.flatten(), nonexceedance) 
    
    s_fdc_max = np.zeros_like(nonexceedance)
    s_fdc_min = np.zeros_like(nonexceedance)
    h_fdc_max = np.zeros_like(nonexceedance)
    h_fdc_min = np.zeros_like(nonexceedance)

    annual_synthetics = Qs.groupby(Qs.index.year)
    annual_historic = Qh.groupby(Qh.index.year)

    for i, quant in enumerate(nonexceedance):
            s_fdc_max[i] = annual_synthetics.quantile(quant).max().max()
            s_fdc_min[i] = annual_synthetics.quantile(quant).min().min()
            h_fdc_max[i] = annual_historic.quantile(quant).max()
            h_fdc_min[i] = annual_historic.quantile(quant).min()
    
    ## Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    #for quant in syn_fdc_quants:
    ax.fill_between(nonexceedance, s_fdc_min, s_fdc_max, color = colors[1], label = 'Synthetic Annual FDC Range', alpha = 0.5)
    ax.fill_between(nonexceedance, h_fdc_min, h_fdc_max, color = colors[0], label = 'Historic Annual FDC Range', alpha = 0.3)

    ax.plot(nonexceedance, s_total_fdc, color = colors[1], label = 'Synthetic Total FDC', alpha = 1)
    ax.plot(nonexceedance, h_total_fdc, color = colors[0], label = 'Historic Total FDC', alpha = 1, linewidth = 2)

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'Flow ({units})')
    ax.set_xlabel('Non-Exceedance Probability')
    ax.legend(fontsize= 10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    plt.title(f'Flow Duration Curve Ranges\nHistoric & Synthetic Streamflow\n{title_addon}')
    if savefig:
        plt.savefig(f'{fig_dir}/flow_duration_curves_{title_addon}.png', dpi=200)
    plt.show()
    return

