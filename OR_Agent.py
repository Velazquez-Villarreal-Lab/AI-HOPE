import sys
import pickle
import matplotlib.pyplot as plt
import pdfkit

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from plotnine import ggplot, aes, geom_bar, labs, theme_bw, annotate , theme, element_text

class OR_Agent:
    def __init__(self):
        self.arg_dict={}

    def chi_square_test(self):

        chi2, self.p_value, dof, expected = stats.chi2_contingency(self.table)
        
        a, b, c, d = self.table.flatten()
        self.odds_ratio = (a * d) / (b * c)
        
        # Confidence interval using statsmodels
        model = sm.stats.Table2x2(self.table)
        self.ci_low, self.ci_high = model.oddsratio_confint()

    def fisher_exact_test(self):
    
        self.odds_ratio, self.p_value = stats.fisher_exact(self.table)
        
        model = sm.stats.Table2x2(self.table)
        self.ci_low, self.ci_high = model.oddsratio_confint()

    def create_report_html(self):

       
        chi2, self.p_value, dof, expected = stats.chi2_contingency(self.table)
        
       
        a, b, c, d = self.table.flatten()
        self.odds_ratio = (a * d) / (b * c)
        
      
        model = sm.stats.Table2x2(self.table)
        self.ci_low, self.ci_high = model.oddsratio_confint()
        
        a, b, c, d = self.table.flatten() 

        
        p_case = round(100*float(a/(a+b)) ,2)
        p_ctrl = round(100*float(c/(c+d)) ,2)
        summary_str = f"Based on the user-defined context, the Odds Ratio Test compares the number of in-context and out-of-context samples between the case and control groups using a 2x2 table with Fisher's exact test. The results are visualized in a stacked bar plot. In summary, there are {a} in-context samples and {b} out-of-context samples in the case group, and {c} in-context samples and {d} out-of-context samples in the control group. The percentage of in-context samples in the case group is {p_case}%, and in the control group is {p_ctrl}%. The p-value from the Chi-square test is {round(self.p_value,3)}. The odd ratio is {round(self.odds_ratio,3)} and the confidence interval (CI) for the odds ratio is [{round(self.ci_low,3)}, {round(self.ci_high,3)}], indicating the significance of the comparison."
        
        logic_str = "You didn't set up any criteria to subset your data."
        if len(self.arg_dict["criteria_logic"])>0:
            logic_str = "The context is based on the logic expression you provided:   "+ str(self.arg_dict["criteria_str"]) +", where each letter represents an individual clause as follows: "+ str(self.arg_dict["criteria_logic"])
        
        png_fname = self.arg_dict["output_png"]
        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
  
        <h3>Comparison of In-Context and Out-of-Context Samples in Case and Control Groups</h3>
        <center> 
        <img src={png_fname} width="450" height="450"  class="center" >
        </center> 
        <h4> {summary_str}</h4>
        <h4> {logic_str}</h4>         
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            f.write(out_str)
        f.close()
        
        pdfkit.from_string(out_str,  self.arg_dict["output_pdf"]  ,options={"enable-local-file-access": ""})




    def plot(self, filename ):

        a, b, c, d = self.table.flatten()
        data = {
            'Group': ['Case', 'Case', 'Control', 'Control'],
            'Outcome': ['In_Context', 'Out_of_Context', 'In_Context', 'Out_of_Context'],
            'Count': [a, b, c, d]
        }
        df = pd.DataFrame(data)

        plot = (
            ggplot(df, aes(x='Group', y='Count', fill='Outcome')) 
            + geom_bar(stat='identity')   
            + labs(x='Group', y='Count', title='Stacked Bar Plot of 2x2 Table') 
            + theme_bw()   
            + theme(
                plot_title=element_text(size=16),
                axis_title_x=element_text(size=16),
                axis_title_y=element_text(size=16),
                legend_title=element_text(size=16),
                legend_text=element_text(size=16)
            )
        )

        plot.save(filename, width=6, height=6)
    def run(self, arg_fname):
        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict

        self.table = np.array([[float(self.arg_dict["Case_in"]), float(self.arg_dict["Case_out"])], [float(self.arg_dict["Ctrl_in"]), float(self.arg_dict["Ctrl_out"])]])
        self.plot(self.arg_dict["output_png"])
        self.create_report_html()

if __name__ == "__main__":

    agent = OR_Agent()
    agent.run(sys.argv[1]) 