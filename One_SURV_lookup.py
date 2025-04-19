import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
import warnings

class One_SURV_lookup:
    def __init__(self, arg_fname):
        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)
        self.filepath = self.arg_dict["Case_metafname"]
        self.variable_name =  self.arg_dict["associated_attr"]
        self.df = self._load_data()
        self.Sample_ID = self.arg_dict["Case_ID"]
        self.is_numeric = pd.api.types.is_numeric_dtype(self.df[self.variable_name])
        self.output_dir= self.arg_dict["output_path"]
        self.results = []
        self.subset_dataframe()
    
    def subset_dataframe(self):
        if len(self.Sample_ID) >0 :
            self.df.index = self.df.index.astype(str)
            self.df = self.df[self.df.index.isin(self.Sample_ID)]
            print(f"Subsetted dataframe to {len(self.df)} samples based on Sample_ID list.")


    def _load_data(self):
        if self.filepath.endswith('.tsv'):
            df = pd.read_csv(self.filepath , sep="\t", index_col=0,  header=0 ,na_values=["none", ""]) 
        else:
            raise ValueError("Only TSV files are supported in this version.")
        
        required_cols = [self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"],self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"]]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.dropna(subset=required_cols + [self.variable_name]).copy()

        return df

    def run_analysis(self):
        # for time_col, event_col, label in [
        #     (self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"], 'Overall Survival'),
        #     (self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"], 'Progression-Free Survival')
        # ]:
        self._run_cox_model(self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"], 'Overall Survival', self.arg_dict["output_forest_OS_png"])
        self._plot_km_curve(self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"], 'Overall Survival',self.arg_dict["output_KM_OS_png"])
        self._run_cox_model(self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"], 'Progression-Free Survival', self.arg_dict["output_forest_PFS_png"])
        self._plot_km_curve(self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"], 'Progression-Free Survival',self.arg_dict["output_KM_PFS_png"])
        
        self.results = pd.concat(self.results, ignore_index=True)
        self.create_report_html()

    def _run_cox_model(self, time_col, event_col, label,out_png_fname):
        col = self.variable_name
        temp_df = self.df[[time_col, event_col, col]].dropna()

        temp_df = temp_df.rename(columns={time_col: 'duration', event_col: 'event'})
        # temp_df = self.df[[time_col, event_col, col]].dropna()

        if not self.is_numeric:
            value_counts = temp_df[col].value_counts()
            valid_categories = value_counts[value_counts >= 5].index
            temp_df = temp_df[temp_df[col].isin(valid_categories)]
            temp_df = pd.get_dummies(temp_df, columns=[col], drop_first=True, prefix=col)

            # print("cow bae")
            # cph_df[self.variable_name] = cph_df[self.variable_name].astype('category')
        cph_df = temp_df
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(cph_df, duration_col='duration', event_col='event')

        summary = cph.summary.reset_index()
        summary['Survival_Type'] = label
        self.results.append(summary)

        self._plot_forest(summary, label, out_png_fname)

    def _plot_forest(self, summary_df, label, out_png_fname):
        df = summary_df.copy()
        df['HR'] = np.exp(df['coef'])
        df['Lower_CI'] = np.exp(df['coef lower 95%'])
        df['Upper_CI'] = np.exp(df['coef upper 95%'])

        plt.figure(figsize=(6, len(df) * 0.6 + 1))
        plt.errorbar(df['HR'], df.index, 
                     xerr=[df['HR'] - df['Lower_CI'], df['Upper_CI'] - df['HR']], 
                     fmt='o', capsize=4)
        plt.axvline(x=1, color='grey', linestyle='--')
        plt.yticks(df.index, df['covariate'])
        plt.xlabel('Hazard Ratio (HR)')
        plt.title(f'Forest Plot: {label}')
        plt.tight_layout()

        filename = out_png_fname 
        plt.savefig(filename, dpi=300)
        plt.close()
        

    def _plot_km_curve(self, time_col, event_col, label, out_png_fname):
        from lifelines.statistics import logrank_test, multivariate_logrank_test
        from lifelines.plotting import add_at_risk_counts

        df = self.df.copy()
        cutoff_label = ""

        if self.is_numeric:
            tertiles = np.quantile(df[self.variable_name], [1/3, 2/3])
            t1, t2 = tertiles[0], tertiles[1]

            def tertile_group(val):
                if val <= t1:
                    return f'Low (≤ {t1:.2f})'
                elif val <= t2:
                    return f'Medium ({t1:.2f} < x ≤ {t2:.2f})'
                else:
                    return f'High (> {t2:.2f})'
            df['group'] = df[self.variable_name].apply(tertile_group)
            cutoff_label = f"Tertiles: Low ≤ {t1:.2f}, Medium ≤ {t2:.2f}, High > {t2:.2f}"
        else:
            df['group'] = df[self.variable_name]

        fig, ax = plt.subplots(figsize=(8, 6))
        group_kmfs = {}

        for name, grouped_df in df.groupby('group'):
            kmf = KaplanMeierFitter()
            kmf.fit(grouped_df[time_col], grouped_df[event_col], label=str(name))
            group_kmfs[name] = kmf
            kmf.plot_survival_function(ax=ax, ci_show=False)

        # Log-rank test
        p_value = None
        if len(group_kmfs) >= 2:
            if len(group_kmfs) == 2:
                g1, g2 = list(group_kmfs.keys())
                df1 = df[df['group'] == g1]
                df2 = df[df['group'] == g2]
                results = logrank_test(df1[time_col], df2[time_col],
                                       event_observed_A=df1[event_col],
                                       event_observed_B=df2[event_col])
                p_value = results.p_value
            else:
                results = multivariate_logrank_test(df[time_col], df['group'], df[event_col])
                p_value = results.p_value

        if p_value is not None:
            ax.text(0.7, 0.05, f'p = {p_value:.3e}', transform=ax.transAxes)

        ax.set_title(f"KM Plot: {label} by {self.variable_name}\n{cutoff_label}", fontsize=10)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')

        # Add number at risk table below the KM plot
        
        add_at_risk_counts(*group_kmfs.values(), ax=ax, rows_to_show=['At risk'])

        plt.tight_layout()
        filename = out_png_fname 
        plt.savefig(filename, dpi=300)
        plt.close()
        

    def create_report_html(self ):
        
        OS_Surv_str = ""
        PFS_Surv_str = ""
        COX_str=""  
        
        if True :
            OS_Surv_str = f"""
            
            <h3>Overall Survival Analyis</h3>
            <center> 
            <img src={self.arg_dict["output_KM_OS_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> Kaplan-Meier Plot for Overall Survival Stratified by User-Defined Context. The plot compares survival probabilities across groups defined by {self.variable_name}. The x-axis shows time, and the y-axis shows survival probability. Statistical significance between curves is assessed using the log-rank test, where a p-value <0.05 suggests the user context may significantly impact survival outcomes.</h4>
            <center> 
            <img src={self.arg_dict["output_forest_OS_png"]} width="450" height="260"  class="center" >
            </center> 
            <h4> Forest Plot of Hazard Ratios from Cox Proportional Hazards Model. The plot shows hazard ratios (HRs) for covariates, with points representing HRs and horizontal lines indicating 95% confidence intervals (CIs). The red vertical line at HR = 1 serves as a reference, where HR > 1 suggests increased risk and HR < 1 suggests decreased risk. CIs crossing the line indicate non-significant effects. This plot summarizes each covariate's impact on survival outcomes. </h4>
            
            <h3>Progression Survival Analyis</h3>
            <center> 
            <img src={self.arg_dict["output_KM_PFS_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> Kaplan-Meier Plot for Progression-Free Survival Stratified by User-Defined Context. The plot compares survival probabilities across groups defined by {self.variable_name}. The x-axis shows time, and the y-axis shows survival probability. Statistical significance between curves is assessed using the log-rank test, where a p-value <0.05 suggests the user context may significantly impact survival outcomes.</h4>
            <center> 
            <img src={self.arg_dict["output_forest_PFS_png"]} width="450" height="260"  class="center" >
            </center> 
            <h4> Forest Plot of Hazard Ratios from Cox Proportional Hazards Model. The plot shows hazard ratios (HRs) for covariates, with points representing HRs and horizontal lines indicating 95% confidence intervals (CIs). The red vertical line at HR = 1 serves as a reference, where HR > 1 suggests increased risk and HR < 1 suggests decreased risk. CIs crossing the line indicate non-significant effects. This plot summarizes each covariate's impact on survival outcomes. </h4>
            
            """

        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
        {OS_Surv_str}
             
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()

if __name__ == '__main__':
   

    surv = One_SURV_lookup(sys.argv[1])
    results_df = surv.run_analysis()
    # results_df.to_csv('cox_results.tsv', sep='\t', index=False)
