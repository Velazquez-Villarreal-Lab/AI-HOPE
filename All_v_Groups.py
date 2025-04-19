import sys
import pickle
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu, kruskal

class All_v_Groups:
    def __init__(self,arg_fname):

        categorical_threshold=10
        self.results = []
        self.categorical_threshold = categorical_threshold
        
        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)

        
        self.df = self.subset_and_merge( self.arg_dict["Case_metafname"],  self.arg_dict["Ctrl_metafname"],self.arg_dict["Case_ID"],self.arg_dict["Ctrl_ID"])
        self.VOI_name = "user_context"

   

    def subset_and_merge(self,Case_metafname, Ctrl_metafname, Case_ID, Ctrl_ID):
    
        # Load data
        case_df = pd.read_csv(Case_metafname , sep="\t", index_col=0,  header=0 ,na_values=["none", ""])  
        ctrl_df = pd.read_csv(Ctrl_metafname , sep="\t", index_col=0,  header=0 ,na_values=["none", ""]) 

        # Subset by provided IDs
        case_subset = case_df.loc[case_df.index.intersection(Case_ID)]
        ctrl_subset = ctrl_df.loc[ctrl_df.index.intersection(Ctrl_ID)]

        # Find common columns
        common_cols = list(case_subset.columns.intersection(ctrl_subset.columns))

        # Merge by rows using only the common columns
        merged_df = pd.concat([
            case_subset[common_cols],
            ctrl_subset[common_cols]
        ])
        merged_df['user_context'] = 0
    
        merged_df.loc[Case_ID, "user_context"] = 1

        return merged_df

    def is_categorical(self, series):
        return series.dtype == "object" or series.nunique() <= self.categorical_threshold

    def run(self):

        clinical_cols = [col for col in  self.df.columns
                         if not (col.startswith('OS_') or col.startswith('PFS_') or col.startswith('DFS_') or col.startswith('DSS_'))]
        for col in clinical_cols:
            if col == self.VOI_name:
                continue

            attr = self.df[col].dropna()
            voi = self.df[self.VOI_name].loc[attr.index].dropna()
            attr = attr.loc[voi.index]
            if attr.empty or voi.empty:
                continue

            is_attr_cat = self.is_categorical(attr)
            is_voi_cat = self.is_categorical(voi)

            try:
                if is_attr_cat and is_voi_cat:
                    contingency_table = pd.crosstab(attr, voi)
                    _, p, _, _ = chi2_contingency(contingency_table)
                    test_type = "Chi-square"
                    stat = None
                elif is_attr_cat != is_voi_cat:
                    if is_attr_cat:
                        groups = [voi[attr == val] for val in attr.unique()]
                    else:
                        groups = [attr[voi == val] for val in voi.unique()]

                    if len(groups) == 2:
                        stat, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                        test_type = "Mann-Whitney U"
                    else:
                        stat, p = kruskal(*groups)
                        test_type = "Kruskal-Wallis"
                else:
                    stat, p = spearmanr(attr, voi)
                    test_type = "Spearman Correlation"

                self.results.append({
                    "Attribute": col,
                    "Test": test_type,
                    "Statistic": stat,
                    "P-value": p
                })
            except Exception as e:
                self.results.append({
                    "Attribute": col,
                    "Test": "Error",
                    "Statistic": None,
                    "P-value": None,
                    "Error": str(e)
                })

        result_df = pd.DataFrame(self.results)
        result_df = result_df.sort_values(by='P-value')
        result_df.to_csv(self.arg_dict["output_tsv"] , sep='\t', index=False)
        self.create_report_html()
        return result_df

    def create_report_html(self):
        result_df = pd.DataFrame(self.results)
        significant_categorical = result_df[
            (result_df["Test"].isin(["Chi-square"])) &
            (result_df["P-value"] < 0.05)
        ]
        significant_categorical = significant_categorical[["Attribute"]]
        print(significant_categorical)
        sc_str = significant_categorical.to_html(classes="table table-bordered", border=0.1, index=False) 
        # Filter for numerical attributes with Spearman's rho > 0.3
        significant_spearman = result_df[
            (result_df["Test"] == "Mann-Whitney U") &
            (result_df["Statistic"].abs() > 0.3) &
            (result_df["P-value"] < 0.05)
        ]
        print(significant_spearman.shape)
        significant_spearman = significant_spearman[["Attribute"]]
        sn_str = significant_spearman.to_html(classes="table table-bordered", border=0.1, index=False) 

        out_str = f"""
            <h3>Association Analysis Between the Case and Control Groups</h3>
            <h4> To assess the relationship between the case-control grouping and other clinical attributes, the analysis automatically selected the appropriate statistical test based on the data types. </h4>
            <h4> The case and control groups are defined as follows:</h4> 
            <h4> Case:   </h4> 
            <h4> {self.arg_dict["Case_criteria_str"]} </h4>
            <h4> {self.arg_dict["Case_criteria_logic"]} </h4>
            <h4> Control: </h4>
            <h4> {self.arg_dict["Case_criteria_str"]} </h4>
            <h4> {self.arg_dict["Case_criteria_logic"]} </h4>
            
            <h3>Categorical Attributes Associated with the Case and Control Groups</h3>
            
            <h4>Based on your inquiry, the following clinical attributes demonstrated meaningful associations with the case and control groups (p-value < 0.05).</h4>
            <center> 
            <table>
            <tr align= "left" > {sc_str} </tr>  
            </table> 
             
            </center>
            
            <h3>Numerical Attributes Associated with the Case and Control Groups</h3> 
            <h4> the following clinical attributes showed meaningful associations with a p-value less than 0.05.</h4>
            <center> 
            <table>
            <tr align= "left" > {sn_str} </tr>  
            </table> 
            </center> 
            <h3>Output File Description</h3>
            <h4>The comprehensive association results are saved to {self.arg_dict["output_tsv"]}. </h4> 
            <h4> Each row in the file represents a clinical attribute that was tested. The columns include the attribute name, the statistical test used, the test statistic (if applicable), the p-value, and any error messages for attributes that failed to compute. This output offers a comprehensive summary of which clinical variables are significantly associated with the case and control groups and the methods by which these associations were identified. It serves as a valuable resource for downstream interpretation and feature selection in predictive modeling.</h4>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            f.write(out_str)
        f.close()
if __name__ == '__main__':
   

    analyzer = All_v_Groups(sys.argv[1])
    result_df = analyzer.run()