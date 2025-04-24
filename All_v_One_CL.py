import sys
import pickle
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu, kruskal

class All_v_One_CL:
    def __init__(self,arg_fname):
        categorical_threshold=10
        self.results = []
        self.categorical_threshold = categorical_threshold
        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)

        self.filepath = self.arg_dict["Case_metafname"]
        self.df = self._load_data()
        self.VOI_name = self.arg_dict["voi_name"]
        self.Sample_ID = self.arg_dict["Case_ID"]
        # self.df = self.df.dropna()
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
        return df

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
            (result_df["Test"].isin(["Chi-square", "Mann-Whitney U", "Kruskal-Wallis"])) &
            (result_df["P-value"] < 0.05)
        ]
        significant_categorical = significant_categorical[["Attribute"]]
        print(significant_categorical)
        sc_str = significant_categorical.to_html(classes="table table-bordered", border=0.1, index=False) 
        # Filter for numerical attributes with Spearman's rho > 0.3
        significant_spearman = result_df[
            (result_df["Test"] == "Spearman Correlation") &
            (result_df["Statistic"].abs() > 0.3) &
            (result_df["P-value"] < 0.05)
        ]
        print(significant_spearman.shape)
        significant_spearman = significant_spearman[["Attribute"]]
        sn_str = significant_spearman.to_html(classes="table table-bordered", border=0.1, index=False) 

        out_str = f"""
            <h3>Association Analysis Between Clinical Variables and Variable of Interest ({self.VOI_name})</h3>
            
            <h4> To assess the relationship between the variable of interest (VOI) and other clinical attributes, the analysis automatically selected the appropriate statistical test based on the data types. For two categorical variables, a Chi-square test was used. When one variable was categorical and the other numerical, either a Mann-Whitney U test (for two groups) or Kruskal-Wallis test (for multiple groups) was applied. For two numerical variables, Spearman correlation was used to measure the strength and direction of association. Attributes with p-values less than 0.05 were considered significant for categorical tests, while numerical associations were considered meaningful if Spearman's rho was greater than 0.3 and p-value less than 0.05.</h4>
            
            <h3>Categorical Attributes Associated with {self.VOI_name}</h3>
            
            <h4>Based on your inquiry, the following clinical attributes demonstrated meaningful associations with {self.VOI_name} (p-value < 0.05).</h4>
            <center> 
            <table>
            <tr align= "left" > {sc_str} </tr>  
            </table> 
             
            </center>
            <h3>Numerical Attributes Associated with {self.VOI_name}</h3> 
            <h4> the following clinical attributes showed meaningful associations with the variable of interest, defined as having a Spearman's rho greater than 0.3 and a p-value less than 0.05.</h4>
            <center> 
            <table>
            <tr align= "left" > {sn_str} </tr>  
            </table> 
            </center> 
            <h3>Output File Description</h3>
            <h4>The comprehensive association results are saved to {self.arg_dict["output_tsv"]}. </h4> 
            <h4> Each row in the file corresponds to a clinical attribute tested against the VOI. The columns include the attribute name, the statistical test used, the test statistic (if applicable), the p-value, and any error messages for attributes that failed to compute. This output provides a comprehensive summary of which clinical variables are significantly associated with the VOI and by what method, serving as a useful resource for downstream interpretation or feature selection in predictive modeling.</h4>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            f.write(out_str)
        f.close()
if __name__ == '__main__':
   

    analyzer = All_v_One_CL(sys.argv[1])
    result_df = analyzer.run()
  