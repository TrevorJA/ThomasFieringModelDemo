import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def calculate_lower_bound(df):
    """
    Calculate the lower bound of a time series (tau)
    """
    qmax = df.max()
    qmin = df.min()
    qmedian = df.median()
    tau = (qmax*qmin - qmedian**2) / (qmax + qmin - 2*qmedian)
    tau = tau if tau > 0 else 0
    return tau


def stedinger_normalization(df):
    """
    Normalize a time series using normalization technique in 
    Stedinger and Taylor (1982).
    """
    # Get lower bound; \hat{\tau} for each month
    norm_df = df.copy()
    tau_monthly = df.groupby(df.index.month).apply(calculate_lower_bound)
    for i in range(1, 13):
        tau = tau_monthly[i]
        
        # Normalize
        norm_df[df.index.month == i] = np.log(df[df.index.month == i] - tau)
    return norm_df, tau_monthly

def inverse_stedinger_normalization(df, tau_monthly):
    """
    Inverse normalize a time series using normalization technique in 
    Stedinger and Taylor (1982).
    """
    # Get lower bound; \hat{\tau} for each month
    norm_df = df.copy()
    for i in range(1, 13):
        tau = tau_monthly[i]
        
        # Normalize
        norm_df[df.index.month == i] = np.exp(df[df.index.month == i]) + tau
    
    return norm_df

class ThomasFieringGenerator():
    """
    From Thomas and Fiering, 1962.    
    Also described in Steinder and Taylor (1982).
    
    Usage:
    tf = ThomasFieringGenerator(Q_obs_monthly.iloc[:,2])
    Q_syn = tf.generate(n_years=10, n_realizations=10)
    """
    def __init__(self, Q, **kwargs):
        # Q should be pandas df or series with monthly index
        if Q.index.freq not in ['MS', 'M']:
            if Q.index.freq in ['D', 'W']:
                Q = Q.resample('MS').sum()
        
        self.Q_obs_monthly = Q 
        
        self.is_fit = False
        self.is_preprocessed = False
    
    def preprocessing(self, **kwargs):
        self.Q_norm, self.tau_monthly = stedinger_normalization(self.Q_obs_monthly)
        self.is_preprocessed = True
        return
        
    def fit(self, **kwargs):
        
        # monthly mean and std
        self.mu = self.Q_norm.groupby(self.Q_norm.index.month).mean()
        self.sigma = self.Q_norm.groupby(self.Q_norm.index.month).std()
        
        # monthly correlation between month m and m+1
        self.rho = self.mu.copy()
        for i in range(1, 13):
            first_month = i
            second_month = i+1 if i < 12 else 1
            first_month_flows = self.Q_norm[self.Q_norm.index.month == first_month]
            second_month_flows = self.Q_norm[self.Q_norm.index.month == second_month]
            if len(first_month_flows) > len(second_month_flows):
                first_month_flows = first_month_flows.iloc[1:]
            elif len(first_month_flows) < len(second_month_flows):
                second_month_flows = second_month_flows.iloc[:-1]
            lag1_r = pearsonr(first_month_flows.values, 
                              second_month_flows.values)
            
            self.rho[i] = lag1_r[0]
            
        self.is_fit = True
    
    def generate(self, 
                 n_years, 
                 **kwargs):
            
        # Generate synthetic sequences
        self.x_syn = np.zeros((n_years*12))
        for i in range(n_years):
            for m in range(12):
                prev_month = m if m > 0 else 12
                month = m + 1
                
                if (i==0) and (m==0):
                    self.x_syn[0] = self.mu[month] + np.random.normal(0, 1)*self.sigma[month]

                else:
                    
                    e_rand = np.random.normal(0, 1)
                    
                    self.x_syn[i*12+m] = self.mu[month] + \
                                        self.rho[month]*(self.sigma[month]/self.sigma[prev_month])*\
                                            (self.x_syn[i*12+m-1] - self.mu[prev_month]) + \
                                                np.sqrt(1-self.rho[month]**2)*self.sigma[month]*e_rand
                        
        # convert to df
        syn_start_year = self.Q_obs_monthly.index[0].year
        syn_start_date = f'{syn_start_year}-01-01'
        self.x_syn = pd.DataFrame(self.x_syn, 
                                index=pd.date_range(start=syn_start_date, 
                                                    periods=len(self.x_syn), freq='MS'))
        self.x_syn[self.x_syn < 0] = self.Q_norm.min()
        
        self.Q_syn = inverse_stedinger_normalization(self.x_syn, self.tau_monthly)
        self.Q_syn[self.Q_syn < 0] = self.Q_obs_monthly.min()
        return self.Q_syn
    
    def generate_ensemble(self, 
                          n_years, 
                          n_realizations = 1, 
                          **kwargs):
        
        if not self.is_preprocessed:
            self.preprocessing()
        if not self.is_fit:
            self.fit()
        
        # Generate synthetic sequences
        for i in range(n_realizations):
            Q_syn = self.generate(n_years)
            if i == 0:
                self.Q_syn_df = Q_syn
            else:
                self.Q_syn_df = pd.concat([self.Q_syn_df, Q_syn], 
                                          axis=1, ignore_index=True)
    
        return self.Q_syn_df