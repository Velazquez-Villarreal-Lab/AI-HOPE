import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdfkit
#### install wkhtmltopdf

class DS_Agent:
    def __init__(self):
        
        self.arg_dict={}
        self.metadata_df =""
        self.n_num = 0
        self.max_ylabel = 0
    def create_report_html(self):
        df = self.metadata_df
        png_fname = self.arg_dict["output_png"]
        height_str= 450
        attr = self.arg_dict['Attr_ID']
        summary_html =""
        if self.n_num == 0:
            summary_df = pd.DataFrame({
                'Mean': [df[ attr].mean()],
                'Median': [df[ attr].median()],
                'Standard Deviation': [df[attr].std()],
                'Minimum': [df[attr].min()],
                'Maximum': [df[attr].max()],
                '25th Percentile': [df[attr].quantile(0.25)],
                '50th Percentile (Median)': [df[attr].quantile(0.5)],
                '75th Percentile': [df[attr].quantile(0.75)]
            })
            summary_df = summary_df.round(2)
            summary_df = summary_df.T
            summary_df.columns = ['Value'] 
            summary_html = summary_df.to_html(classes="table table-bordered", border=0.1, index=True)
        if self.n_num >0 :
             height_str= 360 + 90*self.n_num 
        figure_legend = "Histogram and distribution curve of the numeric column, showing the frequency of values. The smooth curve represents the density estimate, highlighting central clustering and spread of the data. The x-axis denotes values in the numeric column, while the y-axis shows frequency."
        if self.n_num >0 :
             figure_legend= "Bar plots show the distribution of values in the non-numeric column. The plot on the top displays the count of each unique value, while the plot on the bottom shows the percentage representation of each value. The y-axis lists the unique values, and the x-axis represents either the count or percentage."
        
        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
  
        <h3>Distribution of Values for {self.arg_dict['Attr_ID']} </h3>
        <center> 
        <img src={png_fname} width="450" height="{height_str}"  class="center" >
        </center> 
        <h4>{figure_legend}  </h4>
        <center> 
        <h4><tr align= "center" > {summary_html} </tr> </h4>  
        </center> 
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

        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        print(self.arg_dict)
        # in_context = int(self.arg_dict["selected_num"])  # Replace with your actual value
        # out_of_context = int(self.arg_dict["total_num"]) - int(self.arg_dict["selected_num"])  # Replace with your actual value
        attr = self.arg_dict['Attr_ID']
        self.metadata_df = pd.read_csv(self.arg_dict["metafname"], sep="\t", index_col=0,  header=0 ,na_values=["none", ""])

        self.metadata_df = self.metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        print(pd.api.types.is_numeric_dtype(self.metadata_df[attr]))
        if pd.api.types.is_numeric_dtype(self.metadata_df[attr]) and self.metadata_df[attr].nunique() > 2:
            print("numeric")
            plt.figure(figsize=(8, 6))
            sns.histplot(self.metadata_df [attr], kde=True)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram and Distribution Curve of Numeric Column')
            plt.tight_layout()
            plt.savefig(self.arg_dict["output_png"], format="png")

        else:
            value_counts = self.metadata_df[attr].value_counts()
            value_counts.index = value_counts.index.astype(str)
            self.max_ylabel = max(len(s) for s in value_counts.index.to_list() )
            print(self.max_ylabel )
            self.n_num = value_counts.shape[0]
            print( self.n_num)
            value_percentages = (value_counts / len(self.metadata_df)) * 100
            plt.figure(figsize=(8, 4+ self.n_num))
            plt.subplot(2, 1, 1)  # 1 row, 2 columns, first subplot
            value_counts.plot(kind='barh')
            plt.title('Counts of Each Value')
            plt.xlabel('Count')
            plt.ylabel('Value')
            for index, value in enumerate(value_counts):
                plt.text(value, index, str(value), va='center')
# Create horizontal bar plot for percentages
            plt.subplot(2, 1, 2)  # 1 row, 2 columns, second subplot
            value_percentages.plot(kind='barh', color='orange')
            plt.title('Percentage of Each Value')
            plt.xlabel('Percentage (%)')
            plt.ylabel('Value')
            for index, value in enumerate(value_percentages):
                plt.text(value, index, f"{value:.1f}%", va='center')  # Format to one decimal place


# Save as PNG
            plt.tight_layout()
            plt.savefig(self.arg_dict["output_png"], format="png")  # dpi=300 for high resolution


        self.create_report_html()
        


      



# Example usage within this script
if __name__ == "__main__":

    agent = DS_Agent()
    agent.run(sys.argv[1])    


