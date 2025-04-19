import sys
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, kruskal

class One_v_One_CL:
    def __init__(self,arg_fname):

        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)
        self.filepath = self.arg_dict["Case_metafname"]
        self.df = self._load_data()
        self.Sample_ID = self.arg_dict["Case_ID"]
        print(self.df)
        self.VOI_name = self.arg_dict["voi_name"]
        print(self.VOI_name)
        self.Asso_name = self.arg_dict["associated_attr"]
        self.VOI_is_cat = self.df[str(self.VOI_name) ].dtype == 'object' or self.df[str(self.VOI_name)].nunique() <= 10
        self.Asso_is_cat = self.df[self.Asso_name].dtype == 'object' or self.df[self.Asso_name].nunique() <= 10
        self.df = self.df[[self.VOI_name, self.Asso_name]].dropna()
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

    def run_analysis(self):
        if self.VOI_is_cat and self.Asso_is_cat:
            self._categorical_vs_categorical()
        elif self.VOI_is_cat != self.Asso_is_cat:
            self._categorical_vs_numerical()
        else:
            self._numerical_vs_numerical()

    def _create_report_html_c_c(self ):

        html_str = f"""
            
            <h3>Automated Statistical Association and Visualization Tool for Clinical Variables</h3>
            <center> 
            <img src={self.arg_dict["output_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> When both variables are categorical, we assesses their association using the Chi-square test of independence, which evaluates whether the distribution of one categorical variable differs across the levels of the other. This test is commonly used for contingency tables and provides a p-value indicating whether there is a statistically significant relationship between the two categorical attributes. To complement the test, we create two stacked bar plots side by side: one showing the raw counts of each category combination and another showing the percentages within each group of the variable of interest. Each bar segment is labeled with the corresponding count or percentage, making the plots easy to interpret.</h4>
            
            """

        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
        {html_str}
             
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()
    def _create_report_html_n_n(self ):

        html_str = f"""
            
            <h3>Automated Statistical Association and Visualization Tool for Clinical Variables</h3>
            <center> 
            <img src={self.arg_dict["output_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> When both the variable of interest and the variable to test are numerical, the program evaluates their association using the Spearman correlation coefficient, a non-parametric method that measures the strength and direction of a monotonic relationship. This approach is robust to outliers and does not assume a linear or normally distributed relationship, making it well-suited for clinical data. To visualize the association, the program generates a scatter plot with the first numerical variable on the x-axis and the second on the y-axis. A regression line is overlaid along with a 95% confidence interval shaded region to indicate the uncertainty around the fitted trend. The plot provides a clear visual impression of the relationship, while the Spearman correlation coefficient and p-value are printed to summarize the statistical strength and significance of the association.</h4>
            
            """

        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
        {html_str}
             
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()

    def _create_report_html_c_n(self ):

        html_str = f"""
            
            <h3>Automated Statistical Association and Visualization Tool for Clinical Variables</h3>
            <center> 
            <img src={self.arg_dict["output_png"]} width="450" height="460"  class="center" >
            </center>
            <h4> When comparing a numerical variable with a categorical variable, we choose between two non-parametric tests depending on the number of groups in the categorical variable. If there are exactly two groups, we use the Mann-Whitney U test; if there are more than two groups, the Kruskal-Wallis H-test is applied. These tests assess whether the distribution of the numerical variable differs across the categories without assuming normality or equal variance. To visualize the results, a boxplot is created, with the categorical variable on the x-axis and the numerical variable on the y-axis. The groups are sorted by the median of the numerical variable from low to high, making trends more visually interpretable. The p-value from the statistical test is included in the plot title, allowing users to see both the visual and statistical outcome at a glance.</h4>
            
            """

        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
        {html_str}
             
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()

    def _categorical_vs_categorical(self):
        print("Both variables are categorical - running Chi-square test.")
        contingency = pd.crosstab(self.df[self.VOI_name], self.df[self.Asso_name])
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"Chi-square p-value: {p:.4e}")

        # Create two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Count plot
        count_plot = contingency.plot(kind='bar', stacked=True, ax=axes[0])
        axes[0].set_title('Count by Category')
        axes[0].set_ylabel('Count')
        for container in count_plot.containers:
            count_plot.bar_label(container, label_type='center', fontsize=8)

        # Percentage plot
        contingency_perc = contingency.div(contingency.sum(axis=1), axis=0)
        perc_plot = contingency_perc.plot(kind='bar', stacked=True, ax=axes[1])
        axes[1].set_title('Percentage by Category')
        axes[1].set_ylabel('Percentage')
        for container in perc_plot.containers:
            perc_plot.bar_label(container, labels=[f"{v.get_height():.1%}" for v in container], label_type='center', fontsize=8)
        plt.title(f"Chi-square p-value: {p:.4e}")
        plt.tight_layout()
        plt.savefig(self.arg_dict["output_png"])
        self._create_report_html_c_c()

    def _categorical_vs_numerical(self):
        print("One variable is categorical and one is numerical - running non-parametric test.")
        cat_var = self.VOI_name if self.VOI_is_cat else self.Asso_name
        num_var = self.Asso_name if self.VOI_is_cat else self.VOI_name

        # Calculate median for sorting
        median_order = self.df.groupby(cat_var)[num_var].median().sort_values().index.tolist()

        unique_groups = self.df[cat_var].unique()
        if len(unique_groups) == 2:
            group1 = self.df[self.df[cat_var] == unique_groups[0]][num_var]
            group2 = self.df[self.df[cat_var] == unique_groups[1]][num_var]
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"Mann-Whitney U p-value: {p:.4e}")
        else:
            groups = [self.df[self.df[cat_var] == group][num_var] for group in unique_groups]
            groups = [g for g in groups if len(g) > 0 and g.nunique() > 1]
            stat, p = kruskal(*groups)
            print(f"Kruskal-Wallis H-test p-value: {p:.4e}")

        plt.figure()
        sns.boxplot(x=cat_var, y=num_var, data=self.df, order=median_order)
        plt.title(f'Boxplot of {num_var} by {cat_var}\n(p = {p:.2e})')
        plt.tight_layout()
        plt.savefig(self.arg_dict["output_png"])
        self._create_report_html_c_n()

    def _numerical_vs_numerical(self):
        print("Both variables are numerical - running Spearman correlation and regression.")
        rho, p = spearmanr(self.df[self.VOI_name], self.df[self.Asso_name])
        print(f"Spearman correlation: rho={rho:.2f}, p-value={p:.4e}")
        
        plt.figure()
        sns.regplot(x=self.VOI_name, y=self.Asso_name, data=self.df, ci=95, scatter_kws={'alpha':0.5})
        plt.title(f"Spearman correlation: rho={rho:.2f}, p-value={p:.4e}")
        plt.tight_layout()
        plt.savefig(self.arg_dict["output_png"])
        self._create_report_html_n_n()

if __name__ == '__main__':
   

    analyzer = One_v_One_CL(sys.argv[1])

    analyzer.run_analysis()