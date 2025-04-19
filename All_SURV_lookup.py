import sys
import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import os
from statsmodels.stats.multitest import multipletests

class All_SURV_lookup:
    def __init__(self, arg_fname ):

        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)
        self.input_file = self.arg_dict["Case_metafname"]
        
        self.Sample_ID = self.arg_dict["Case_ID"]
        self.df = pd.read_csv(self.arg_dict["Case_metafname"], sep="\t", index_col=0,  header=0 ,na_values=["none", ""]) 
        self.subset_dataframe()

    def subset_dataframe(self):
        if len(self.Sample_ID) >0 :
            self.df.index = self.df.index.astype(str)
            self.df = self.df[self.df.index.isin(self.Sample_ID)]
            print(f"Subsetted dataframe to {len(self.df)} samples based on Sample_ID list.")

    def run_univariate_cox(self, time_col, event_col):
        results = []
        all_attributes = []
        clinical_cols = [col for col in self.df.columns.difference([time_col, event_col])
                         if not (col.startswith('OS_') or col.startswith('PFS_') or col.startswith('DFS_') or col.startswith('DSS_'))]


        for col in clinical_cols:
            all_attributes.append(col)
            temp_df = self.df[[time_col, event_col, col]].dropna()

            if temp_df[col].dtype == 'object' or temp_df[col].dtype.name == 'category':
                value_counts = temp_df[col].value_counts()
                valid_categories = value_counts[value_counts >= 5].index
                temp_df = temp_df[temp_df[col].isin(valid_categories)]
                temp_df = pd.get_dummies(temp_df, columns=[col], drop_first=True, prefix=col)

            try:
                cph = CoxPHFitter()
                cph.fit(temp_df, duration_col=time_col, event_col=event_col)
                summary = cph.summary.copy()
                summary.insert(0, 'full_variable', summary.index)
                summary['attribute'] = col
                summary['category_level'] = summary['full_variable'].apply(lambda x: x.replace(f'{col}_', '') if f'{col}_' in x else '')
                summary['has_result'] = True
                summary.reset_index(drop=True, inplace=True)
                results.append(summary)
            except Exception as e:
                print(f"Could not compute Cox regression for {col}: {e}")
                empty_row = pd.DataFrame({
                    'attribute': [col],
                    'category_level': [''],
                    'coef': [np.nan],
                    'exp(coef)': [np.nan],
                    'se(coef)': [np.nan],
                    'z': [np.nan],
                    'p': [np.nan],
                    'FDR': [np.nan],
                    'has_result': [False]
                })
                results.append(empty_row)

        if results:
            full_df = pd.concat(results, ignore_index=True)
            pvals = full_df['p'].values
            mask = ~np.isnan(pvals)
            adjusted = np.full_like(pvals, np.nan, dtype=np.float64)
            adjusted[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
            full_df['FDR'] = adjusted
            
            return full_df
        else:
            return pd.DataFrame({'attribute': all_attributes})

    def run(self):
        all_results = []
        for outcome in [(self.arg_dict["Case_OS_TIME"], self.arg_dict["Case_OS_STATUS"]), (self.arg_dict["Case_PFS_TIME"], self.arg_dict["Case_PFS_STATUS"])]:
            time_col, event_col = outcome
            if time_col in self.df.columns and event_col in self.df.columns:
                print(f"Running Cox regression for {time_col} / {event_col}")
                result_df = self.run_univariate_cox(time_col, event_col)
                result_df.insert(0, 'endpoint', f"{time_col}_{event_col}")
                all_results.append(result_df)
            else:
                print(f"Columns {time_col} and/or {event_col} not found in the input.")

        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.sort_values(by='p')
            output_file = self.arg_dict["output_tsv"]
            combined_df.to_csv(output_file, sep='\t', index=False)
            html_df  = combined_df.loc[combined_df['p']<0.05]
            html_df = html_df.sort_values(by='p')
            html_df = html_df.round(3)
            html_df = html_df[["full_variable"]]
            html_df.columns = ["Attribute"]
            
            html_df_html = html_df.to_html(classes="table table-bordered", border=0.1, index=False)
             
            self.create_report_html(html_df_html)
            print(f"Combined results saved to {output_file}")
        else:
            print("No Cox regression results were generated.")
    
    def create_report_html(self, html_df):
        out_str = f"""
            
            <h3>Clinical Attributes Significantly Associated with Survival Outcomes (p < 0.05) Identified by Cox Regression Analysis</h3>
            <h4>Based on your inquiry, the following clinical attributes demonstrated statistically significant associations with survival outcomes (p-value < 0.05) in the Cox regression analysis. These attributes may have prognostic relevance and warrant further investigation in the context of overall survival (OS) or progression-free survival (PFS). The significance was determined after adjusting for each variable independently in the Cox proportional hazards model.</h4>
            <center> 
            <table>
            <tr align= "left" > {html_df} </tr>  
            </table> 
            </center> 
            <h4>The comprehensive Cox regression results are saved to {self.arg_dict["output_tsv"]}. </h4> <h4> This file contains the output of univariate Cox proportional hazards models evaluating the association between each clinical attribute and two survival endpoints: overall survival (OS) and progression-free survival (PFS). Each row corresponds to a numeric variable or a specific category of a categorical variable. Key columns include attribute (the original variable name), category_level (the specific level of a categorical attribute, left blank for numeric attributes), and reference (the baseline category used in the comparison for categorical variables). The regression outputs include the log hazard ratio (coef), hazard ratio (exp(coef)), 95% confidence intervals, standard error (se(coef)), z-statistic (z), raw p-value (p), and FDR-adjusted p-value (FDR). The column has_result indicates whether the Cox model was successfully computed for the given variable. </h4>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()
        

    def plot_forest(self, result_df, output_file='forest_plot.png'):
        import matplotlib.pyplot as plt

        sig_df = result_df[(result_df['p'] < 0.05) & result_df['has_result'] == True].copy()
        sig_df = sig_df.sort_values('exp(coef)')
        
        if sig_df.empty:
            print("No significant results with p < 0.05 to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(sig_df))))
        y_pos = np.arange(len(sig_df))
        
        ax.errorbar(sig_df['exp(coef)'], y_pos,
                    xerr=[sig_df['exp(coef)'] - sig_df['exp(coef) lower 95%'],
                          sig_df['exp(coef) upper 95%'] - sig_df['exp(coef)']],
                    fmt='o', color='black', ecolor='gray', capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sig_df['full_variable'])
        ax.axvline(x=1, color='red', linestyle='--')
        ax.set_xlabel('Hazard Ratio (exp(coef))')
        ax.set_title('Forest Plot of Significant Cox Regression Results (p < 0.05)')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Forest plot saved to {output_file}")
if __name__ == '__main__':
   

    lookup = All_SURV_lookup(sys.argv[1])
    lookup.run()
