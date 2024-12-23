import sys
import pickle
import matplotlib.pyplot as plt
import pdfkit
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from sklearn.preprocessing import OneHotEncoder

#### install wkhtmltopdf

class SURV_Agent:
    def __init__(self):
        
        self.arg_dict={}

    def create_report_html(self, OS_summary, PFS_summary):
        
        OS_Surv_str = ""
        PFS_Surv_str = ""
        COX_str=""  
        OS_summary_html =""

        if isinstance(OS_summary, pd.DataFrame):
            OS_summary = OS_summary.round(3)
            OS_summary.columns = ["Hazard Ratio","CI_low (95%)","CI_high (95%)","P value"]
            OS_summary_html = OS_summary.to_html(classes="table table-bordered", border=0.1, index=True)
        if isinstance(PFS_summary, pd.DataFrame):
            PFS_summary = PFS_summary.round(3)
            PFS_summary.columns = ["Hazard Ratio","CI_low (95%)","CI_high (95%)","P value"]
            PFS_summary_html = PFS_summary.to_html(classes="table table-bordered", border=0.1, index=True)

        if self.arg_dict["Case_OS_TIME"] and self.arg_dict["Case_OS_STATUS"] and self.arg_dict["Ctrl_OS_TIME"] and self.arg_dict["Ctrl_OS_STATUS"]:
            OS_Surv_str = f"""
            
            <h3>Overall Survival Analyis</h3>
            <center> 
            <img src={self.arg_dict["output_OS_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> Kaplan-Meier Plot for Overall Survival Stratified by User-Defined Context. The plot compares survival probabilities across groups defined by a user context (e.g., treatment type). The x-axis shows time, and the y-axis shows survival probability. Shaded areas represent the 95% confidence intervals (CIs), indicating the range within which true survival probabilities likely fall. Statistical significance between curves is assessed using the log-rank test, where a p-value <0.05 suggests the user context may significantly impact survival outcomes.</h4>
            <center> 
            <img src={self.arg_dict["output_forest_OS_png"]} width="450" height="260"  class="center" >
            </center> 
            <h4> Forest Plot of Hazard Ratios from Cox Proportional Hazards Model. The plot shows hazard ratios (HRs) for covariates, with points representing HRs and horizontal lines indicating 95% confidence intervals (CIs). The red vertical line at HR = 1 serves as a reference, where HR > 1 suggests increased risk and HR < 1 suggests decreased risk. CIs crossing the line indicate non-significant effects. This plot summarizes each covariate's impact on survival outcomes. </h4>
            <center> 
            <h4> <tr align= "center" > {OS_summary_html} </tr> </h4>
            </center> 
            <h3>Progression Survival Analyis</h3>
            <center> 
            <img src={self.arg_dict["output_PFS_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> Kaplan-Meier Plot for Progression-Free Survival Stratified by User-Defined Context. The plot compares survival probabilities across groups defined by a user context (e.g., treatment type). The x-axis shows time, and the y-axis shows survival probability. Shaded areas represent the 95% confidence intervals (CIs), indicating the range within which true survival probabilities likely fall. Statistical significance between curves is assessed using the log-rank test, where a p-value <0.05 suggests the user context may significantly impact survival outcomes.</h4>
            <center> 
            <img src={self.arg_dict["output_forest_PFS_png"]} width="450" height="260"  class="center" >
            </center> 
            <h4> Forest Plot of Hazard Ratios from Cox Proportional Hazards Model. The plot shows hazard ratios (HRs) for covariates, with points representing HRs and horizontal lines indicating 95% confidence intervals (CIs). The red vertical line at HR = 1 serves as a reference, where HR > 1 suggests increased risk and HR < 1 suggests decreased risk. CIs crossing the line indicate non-significant effects. This plot summarizes each covariate's impact on survival outcomes. </h4>
            <center> 
            <h4> <tr align= "center" > {PFS_summary_html} </tr> </h4>
            </center> 
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
        
        pdfkit.from_string(out_str,  self.arg_dict["output_pdf"]  ,options={"enable-local-file-access": "",  "quiet": ""})


   

    def run(self, arg_fname):
        os_summary = None
        pds_summary =None
        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
       

        Case_df = pd.read_csv(self.arg_dict["Case_metafname"], sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
        Ctrl_df = pd.read_csv(self.arg_dict["Ctrl_metafname"], sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
        
        
        Case_df = Case_df.loc[self.arg_dict["Case_ID"]]
        Ctrl_df = Ctrl_df.loc[self.arg_dict["Ctrl_ID"]]
        

        if self.arg_dict["Case_OS_TIME"] and self.arg_dict["Case_OS_STATUS"] and self.arg_dict["Ctrl_OS_TIME"] and self.arg_dict["Ctrl_OS_STATUS"]:
            Case_data_df = Case_df.loc[:,[self.arg_dict["Case_OS_TIME"] , self.arg_dict["Case_OS_STATUS"] ]]
            Case_data_df.columns =[ "OS_TIME", "OS_STATUS" ]
            
            if self.arg_dict["EXTRA_ATTR"] :
                attr_list = self.arg_dict["EXTRA_ATTR"] 

                for it in attr_list:
                    if pd.api.types.is_numeric_dtype(Case_df[it]):
                        df_encoded = Case_df[it]
                    else:
                    # Perform one-hot encoding
                        encoder = OneHotEncoder(sparse_output = False , drop='first')
                        encoded_array = encoder.fit_transform(Case_df[[it]])
                        df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([it]))
                        df_encoded.index = Case_df.index
                    Case_data_df =  pd.concat([Case_data_df, df_encoded], axis=1  )
            Ctrl_data_df = Ctrl_df.loc[:,[self.arg_dict["Ctrl_OS_TIME"] , self.arg_dict["Ctrl_OS_STATUS"] ]]
            Ctrl_data_df.columns =[ "OS_TIME", "OS_STATUS" ]
            
            if self.arg_dict["EXTRA_ATTR"] :
                attr_list = self.arg_dict["EXTRA_ATTR"] 

                for it in attr_list:
                    if pd.api.types.is_numeric_dtype(Ctrl_df[it]):
                        df_encoded = Ctrl_df[it]
                    else:
                        encoder = OneHotEncoder(sparse_output = False , drop='first')
                        encoded_array = encoder.fit_transform(Ctrl_df[[it]])
                        df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([it]))
                        df_encoded.index = Ctrl_df.index
                    Ctrl_data_df =  pd.concat([Ctrl_data_df, df_encoded], axis=1 )
            
                   
            data = pd.concat([Case_data_df, Ctrl_data_df], axis=0 )
            
            data['user_context'] = 0

        
            data.loc[self.arg_dict["Case_ID"], "user_context"] = 1
            
            data =  data.fillna(0)
            kmf_OS = KaplanMeierFitter()
            plt.figure()

            for context_group in data['user_context'].unique():
                mask = data['user_context'] == context_group
                
                class_name = "Control"
                if context_group ==1 :
                    class_name = "Case"
                kmf_OS.fit(data[mask]['OS_TIME'], event_observed=data[mask]['OS_STATUS'], label=f' {class_name}')
                kmf_OS.plot_survival_function()
            
        
            group_case = data[data['user_context'] == 1]
            group_ctrl = data[data['user_context'] == 0]
            results = logrank_test(group_case['OS_TIME'], group_ctrl['OS_TIME'], event_observed_A=group_case['OS_STATUS'], event_observed_B=group_ctrl['OS_STATUS'])

            p_value = results.p_value
            plt.text(0.1, 0.1, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=12, color='black')
            plt.title("Kaplan-Meier Survival Curve")
            plt.xlabel("Time (Months)")
            plt.ylabel("Survival Probability")

            plt.savefig(self.arg_dict["output_OS_png"], format="png")

            cph = CoxPHFitter()
            cph.fit(data, duration_col='OS_TIME', event_col='OS_STATUS')

            os_summary = cph.summary
            hr = os_summary['exp(coef)']
            ci_lower = os_summary['exp(coef) lower 95%']
            ci_upper = os_summary['exp(coef) upper 95%']
            variables = os_summary.index
            plt.figure(figsize=(12, 4+0.3*data.shape[1]))
            plt.errorbar(hr, variables, xerr=[hr - ci_lower, ci_upper - hr], fmt='o', color='black', ecolor='gray', capsize=5)
            plt.axvline(1, color='red', linestyle='--')  # Reference line at hazard ratio of 1
            plt.ylim(-1, len(variables)) 
            plt.xlabel('Hazard Ratio' ,  fontsize=24)
            plt.ylabel('Covariates', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.subplots_adjust(left=0.4, right=0.9, top=0.8, bottom=0.2)
            plt.title('Forest Plot of Hazard Ratios for Cox Proportional Hazards Model',  fontsize=24)
            plt.savefig(self.arg_dict["output_forest_OS_png"],format="png")  # Save the plot as a PNG file
            
            os_summary = os_summary[  ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"] ]
            
        if self.arg_dict["Case_PFS_TIME"] and self.arg_dict["Case_PFS_STATUS"] and self.arg_dict["Ctrl_PFS_TIME"] and self.arg_dict["Ctrl_PFS_STATUS"]:
            Case_data_df = Case_df.loc[:,[self.arg_dict["Case_PFS_TIME"] , self.arg_dict["Case_PFS_STATUS"] ]]
            Case_data_df.columns =[ "PFS_TIME", "PFS_STATUS" ]
            
            if self.arg_dict["EXTRA_ATTR"] :
                attr_list = self.arg_dict["EXTRA_ATTR"] 

                for it in attr_list:
                    if pd.api.types.is_numeric_dtype(Case_df[it]):
                        df_encoded = Case_df[it]
                    else:
                    # Perform one-hot encoding
                        encoder = OneHotEncoder(sparse_output = False , drop='first')
                        encoded_array = encoder.fit_transform(Case_df[[it]])
                        df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([it]))
                        df_encoded.index = Case_df[it].index
                    Case_data_df =  pd.concat([Case_data_df, df_encoded], axis=1 )

            Ctrl_data_df = Ctrl_df.loc[:,[self.arg_dict["Ctrl_PFS_TIME"] , self.arg_dict["Ctrl_PFS_STATUS"] ]]
            Ctrl_data_df.columns =[ "PFS_TIME", "PFS_STATUS" ]
            
            if self.arg_dict["EXTRA_ATTR"] :
                attr_list = self.arg_dict["EXTRA_ATTR"] 

                for it in attr_list:
                    if pd.api.types.is_numeric_dtype(Ctrl_df[it]):
                        df_encoded = Ctrl_df[it]
                    else:
                        encoder = OneHotEncoder(sparse_output = False , drop='first')
                        encoded_array = encoder.fit_transform(Ctrl_df[[it]])
                        df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([it]))
                        df_encoded.index = Ctrl_df[[it]].index
                    Ctrl_data_df =  pd.concat([Ctrl_data_df, df_encoded], axis=1 )
            
            
       
            data = pd.concat([Case_data_df, Ctrl_data_df], axis=0 )
            data['user_context'] = 0

        
            data.loc[self.arg_dict["Case_ID"], "user_context"] = 1
            data =  data.fillna(0)
            data.to_csv("data_surv.csv")
            kmf_OS = KaplanMeierFitter()
            plt.figure()

            for context_group in data['user_context'].unique():
                mask = data['user_context'] == context_group
                
                class_name = "Control"
                if context_group ==1 :
                    class_name = "Case"
                kmf_OS.fit(data[mask]['PFS_TIME'], event_observed=data[mask]['PFS_STATUS'], label=f' {class_name}')
                kmf_OS.plot_survival_function()
            
        
            group_case = data[data['user_context'] == 1]
            group_ctrl = data[data['user_context'] == 0]
            results = logrank_test(group_case['PFS_TIME'], group_ctrl['PFS_TIME'], event_observed_A=group_case['PFS_STATUS'], event_observed_B=group_ctrl['PFS_STATUS'])

            p_value = results.p_value
            plt.text(0.1, 0.1, f'p-value: {p_value:.4f}', transform=plt.gca().transAxes, fontsize=12, color='black')
            plt.title("Kaplan-Meier Survival Curve")
            plt.xlabel("Time (Months)")
            plt.ylabel("Survival Probability")

            plt.savefig(self.arg_dict["output_PFS_png"], format="png")

            cph = CoxPHFitter()
            cph.fit(data, duration_col='PFS_TIME', event_col='PFS_STATUS')

            PFS_summary = cph.summary
            hr = PFS_summary['exp(coef)']
            ci_lower = PFS_summary['exp(coef) lower 95%']
            ci_upper = PFS_summary['exp(coef) upper 95%']
            variables = PFS_summary.index
            plt.figure(figsize=(12,  4+0.3*data.shape[1]))
            plt.errorbar(hr, variables, xerr=[hr - ci_lower, ci_upper - hr], fmt='o', color='black', ecolor='gray', capsize=5)
            plt.axvline(1, color='red', linestyle='--')  # Reference line at hazard ratio of 1
            plt.ylim(-1, len(variables)) 
            plt.xlabel('Hazard Ratio' ,  fontsize=24)
            plt.ylabel('Covariates', fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.subplots_adjust(left=0.4, right=0.9, top=0.8, bottom=0.2)
            plt.title('Forest Plot of Hazard Ratios for Cox Proportional Hazards Model',  fontsize=24)
            plt.savefig(self.arg_dict["output_forest_PFS_png"],format="png")  # Save the plot as a PNG file
            
            PFS_summary = PFS_summary[  ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', "p"] ]

      
        self.create_report_html(os_summary, PFS_summary)
        


      



if __name__ == "__main__":
    agent = SURV_Agent()
    agent.run(sys.argv[1]) 

