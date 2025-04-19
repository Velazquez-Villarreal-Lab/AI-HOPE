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
        
        self.df = self._load_data()
        self.Sample_ID = self.arg_dict["Case_ID"]
        
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
        
        df = df.dropna(subset=required_cols).copy()

        return df
    
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
            
            <h3>Progression Survival Analyis</h3>
            <center> 
            <img src={self.arg_dict["output_KM_PFS_png"]} width="450" height="460"  class="center" >
            </center>
            
            <h4> The Kaplan-Meier (KM) plot displays the estimated survival probability over time for the cohort under study. Each step downward in the curve represents an event (e.g., death, failure, or another defined endpoint), while horizontal segments indicate periods where no events occurred. Censored observations, which represent individuals who were lost to follow-up or did not experience the event by the end of the study, are typically marked with small ticks on the curve. The y-axis shows the probability of survival, starting at 1.0 (100%) and decreasing as events occur. The accompanying at-risk table below the plot provides the number of individuals still under observation (i.e., "at risk") at each time point. This table helps contextualize the reliability of the survival estimates, as smaller at-risk numbers at later time points can lead to greater uncertainty in the curve.  </h4>
            
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

    def _plot_km_curve(self, time_col, event_col, label,out_png_fname):
        from lifelines.statistics import logrank_test, multivariate_logrank_test
        from lifelines.plotting import add_at_risk_counts
        
        df = self.df
        kmf = KaplanMeierFitter()
    
        # Fit the model
        kmf.fit(durations=df[time_col], event_observed=df[event_col], label=label)
    
        # Plot survival function
        ax = kmf.plot(ci_show=True)
    
        # Add at-risk table
        add_at_risk_counts(kmf, ax=ax, rows_to_show=['At risk'])
    
        plt.title("Kaplan-Meier Survival Curve")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.tight_layout()
        filename = out_png_fname 
        plt.savefig(filename, dpi=300)
        plt.close()

    def run_analysis(self):
        # for time_col, event_col, label in [
        #     (self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"], 'Overall Survival'),
        #     (self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"], 'Progression-Free Survival')
        # ]:
        
        self._plot_km_curve(self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"], 'Overall Survival',self.arg_dict["output_KM_OS_png"])
        self._plot_km_curve(self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"], 'Progression-Free Survival',self.arg_dict["output_KM_PFS_png"])
        
        self.create_report_html()

if __name__ == '__main__':
   

    surv = One_SURV_lookup(sys.argv[1])
    results_df = surv.run_analysis()