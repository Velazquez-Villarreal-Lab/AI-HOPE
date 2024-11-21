import sys
import pickle
import matplotlib.pyplot as plt
import pdfkit
import os
#### install wkhtmltopdf

class DS_Agent:
    def __init__(self):
        
        self.arg_dict={}

    def create_report_html(self):
        in_context_num = int(self.arg_dict["selected_num"])
        total_num = int(self.arg_dict["total_num"])
        p = round(100*float(in_context_num/total_num) ,2)
        selection_str = f"The pie chart illustrates the distribution of selected and unselected samples. There are {in_context_num} selected samples out of a total of {total_num} samples, making up {p}% of the dataset. This visualization helps to clearly understand the proportion of samples in each category."
        logic_str = "You didn't set up any criteria to subset your data."
        if len(self.arg_dict["criteria_logic"])>0:
            logic_str = "The selection is based on the logic expression you provided:   "+ str(self.arg_dict["criteria_str"]) +", where each letter represents an individual clause as follows: "+ str(self.arg_dict["criteria_logic"])
        case_id = self.arg_dict["case_id"]
        # print(logic_str)
        png_fname = self.arg_dict["output_png"]
        out_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <blockquote>        
        <body>
  
        <h3>Distribution of Selected and Unselected Samples in Your {case_id} Cohort</h3>
        <center> 
        <img src={png_fname} width="450" height="460"  class="center" >
        </center> 
        <h4> {selection_str}</h4>
        <h4> {logic_str}</h4>         
        </body>
        </blockquote>
        </html>
        """
        with open( self.arg_dict["output_html"] , "w") as f:
            # Write the string to the file
            f.write(out_str)
        f.close()
        
        pdfkit.from_string(out_str,  self.arg_dict["output_pdf"]  ,options={"enable-local-file-access": "",  "quiet": ""})


    def get_image_file_as_base64_data(self,FILEPATH):
        with open(FILEPATH, 'rb') as image_file:
            return base64.b64encode(image_file.read())

    def run(self, arg_fname):

        with open(arg_fname, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.arg_dict = loaded_dict
        # print(self.arg_dict)

        in_context = int(self.arg_dict["selected_num"])  # Replace with your actual value
        out_of_context = int(self.arg_dict["total_num"]) - int(self.arg_dict["selected_num"])  # Replace with your actual value

        # Pie chart data
        sizes = [in_context, out_of_context]

        # Create the pie chart without labels on the pie slices
        plt.figure(figsize=(6, 6))
        wedges, _, autotexts = plt.pie(sizes, autopct='%1.1f%%', textprops={'fontsize': 18})

        # Add a legend with the labels
        labels = [f'Selected: {in_context}', f'Others: {out_of_context}']
        plt.legend(wedges, labels, loc='upper center', fontsize=18, bbox_to_anchor=(0.5, -0.1), ncol=2)

        # Save the pie chart as a PNG file
        plt.savefig(self.arg_dict["output_png"], bbox_inches='tight')
      
        self.create_report_html()
        


      



# Example usage within this script
if __name__ == "__main__":

    agent = DS_Agent()
    agent.run(sys.argv[1])    


