from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from typing import Annotated
from typing_extensions import TypedDict

import tkinter as tk
import sys
from tkinter import font
from tkinter import messagebox
from tkhtmlview import HTMLScrolledText 

import json
import pandas as pd
import pandas.api.types as ptypes
import re
import os
import string
import ast
import Levenshtein
import time

from datetime import datetime
import pickle

import subprocess

from PIL import Image 

from packaging.version import Version
if Version(Image.__version__) >= Version('10.0.0'):
    Image.ANTIALIAS = Image.LANCZOS


class AgentState(TypedDict):
    
    attributes: list
    messages: Annotated[list, add_messages]

class Supervisor:
    def __init__(self, root, model, local_memory ):
        
        graph = StateGraph(AgentState)
        
        graph.add_node("init_Case", self.init_Case_fun)
        graph.add_node("input_data_Case", self.input_data_Case_fun)
        graph.add_node("load_data_Case", self.load_data_Case_fun)
        graph.add_node("init_query_I_Case", self.init_query_I_Case_fun)
        graph.add_node("parse_query_I_Case", self.parse_query_I_fun)
        graph.add_node("init_set_criteria_Case", self.init_set_criteria_fun)
       
        graph.add_node("overview_Case", self.overview_Case_fun)
        graph.add_node("show_attr_values_Case", self.show_attr_values_Case_fun)
        graph.add_node("set_criteria_Case", self.set_criteria_Case_fun)
        graph.add_node("summary_Case", self.summary_Case_fun)

        graph.add_node("init_Ctrl", self.init_Ctrl_fun)
        graph.add_node("input_data_Ctrl", self.input_data_Ctrl_fun)
        graph.add_node("load_data_Ctrl", self.load_data_Ctrl_fun)
        graph.add_node("init_query_I_Ctrl", self.init_query_I_Ctrl_fun)
        graph.add_node("parse_query_I_Ctrl", self.parse_query_I_fun)
        graph.add_node("init_set_criteria_Ctrl", self.init_set_criteria_fun)

        graph.add_node("overview_Ctrl", self.overview_Ctrl_fun)
        graph.add_node("show_attr_values_Ctrl", self.show_attr_values_Ctrl_fun)
        graph.add_node("set_criteria_Ctrl", self.set_criteria_Ctrl_fun)
        graph.add_node("summary_Ctrl", self.summary_Ctrl_fun)

        graph.add_node("init_exec", self.init_exec_fun)
        graph.add_node("parse_exec", self.parse_exec_fun)
        
        graph.add_node("init_OR", self.init_OR_fun)
        graph.add_node("parse_OR", self.parse_OR_fun)

        graph.add_node("init_Survival", self.init_Survival_fun)
        graph.add_node("parse_Survival", self.parse_Survival_fun)
        graph.add_node("init_multiple_Survival", self.init_multiple_Survival_fun)
        graph.add_node("multiple_Survival", self.multiple_Survival_fun)
        graph.add_node("run_Survival", self.run_Survival_fun)
        
        graph.add_edge(START, "init_Case")
        graph.add_edge("init_Case", "input_data_Case")
        

        graph.add_conditional_edges(
            "input_data_Case",
            self.make_decision_fun,
            {1: "load_data_Case", 2:"input_data_Case" }
        )

        graph.add_edge("load_data_Case", "init_query_I_Case")
        graph.add_edge("init_query_I_Case", "parse_query_I_Case")
        graph.add_conditional_edges(
            "parse_query_I_Case",
            self.make_decision_fun,
            {1:"overview_Case", 2: "init_set_criteria_Case",3:"summary_Case" ,4: "init_query_I_Case"}
        )
        
        graph.add_edge("init_set_criteria_Case", "set_criteria_Case")
        graph.add_edge("set_criteria_Case", "init_query_I_Case")

        graph.add_edge("overview_Case", "show_attr_values_Case")
        graph.add_edge("show_attr_values_Case", "init_query_I_Case")
        
        graph.add_edge("summary_Case", "init_Ctrl")

        graph.add_edge("init_Ctrl", "input_data_Ctrl")
       

        graph.add_conditional_edges(
            "input_data_Ctrl",
            self.make_decision_fun,
            {1: "load_data_Ctrl",2:"input_data_Ctrl" }
        )

        graph.add_edge("load_data_Ctrl", "init_query_I_Ctrl")
        graph.add_edge("init_query_I_Ctrl", "parse_query_I_Ctrl")

        graph.add_conditional_edges(
            "parse_query_I_Ctrl",
            self.make_decision_fun,
            {1:"overview_Ctrl", 2: "init_set_criteria_Ctrl",  3:"summary_Ctrl" , 4: "init_query_I_Ctrl"}
        )
        
        graph.add_edge("init_set_criteria_Ctrl", "set_criteria_Ctrl")
        graph.add_edge("set_criteria_Ctrl", "init_query_I_Ctrl")

        graph.add_edge("overview_Ctrl", "show_attr_values_Ctrl")
        graph.add_edge("show_attr_values_Ctrl", "init_query_I_Ctrl")
                
        
        graph.add_conditional_edges(
            "summary_Ctrl",
            self.make_decision_fun,
            {1: "init_exec", 2:"init_Case" }
        )

        graph.add_edge("init_exec", "parse_exec")
        graph.add_conditional_edges(
            "parse_exec",
            self.make_decision_fun,
            {1:"init_OR", 2:"init_Survival", 3: "init_exec"}
        )
        graph.add_edge("init_OR", "parse_OR")
        graph.add_edge("parse_OR", "init_exec")

        graph.add_edge("init_Survival", "parse_Survival")
        graph.add_conditional_edges(
            "parse_Survival",
            self.make_decision_fun,
            { 1:"run_Survival", 2:"init_multiple_Survival",3:"init_Survival"}
        )
        
        graph.add_edge("init_multiple_Survival","multiple_Survival")
        graph.add_edge("multiple_Survival", "run_Survival")
        graph.add_edge("run_Survival", "init_exec")

        self.graph = graph.compile(
            checkpointer=local_memory,
            interrupt_before=["input_data_Case","input_data_Ctrl"  , "parse_query_I_Case","parse_query_I_Ctrl", "show_attr_values_Case","show_attr_values_Ctrl" , "set_criteria_Case","set_criteria_Ctrl", "parse_exec","parse_OR", "parse_Survival", "multiple_Survival" ]
        )
        
        self.model = model
        self.conversation_buffer =[]

        self.data_repository= []

        self.Case_data_id = ""
        self.Case_criteria_str = ""
        self.Case_criteria_logic = {}
        self.Case_sample_ids = []
        self.Case_config_dict ={}
        self.Case_metafname=""
        self.Case_metadata_df = "" 

        self.Ctrl_data_id = ""
        self.Ctrl_criteria_str = ""
        self.Ctrl_criteria_logic = {}
        self.Ctrl_sample_ids = []
        self.Ctrl_config_dict = {}
        self.Ctrl_metafname=""
        self.Ctrl_metadata_df = ""

        self.or_num=1
        self.case_exhibit_num=1
        self.ctrl_exhibit_num=1
        
        self.case_DS_num=1
        self.ctrl_DS_num=1

        self.surv_num=1
        self.surv_extra=[]

        self.root = root
        self.user_input = tk.StringVar()
        self.html_fname = "dialogs/welcome_1.html"
        
        self.root.protocol("WM_DELETE_WINDOW",  self.on_close)
        self.conversation_path = ""
        self.create_widgets()

    def create_widgets(self):
        widget_font = font.Font(family="Helvetica", size=16)
        
        label_font = font.Font(family="Helvetica", size=20)

        output_label = tk.Label(self.root, text="Conversation:", font=label_font)
        output_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.text_frame = tk.Frame(self.root)
        self.text_frame.grid(row=1, column=0, columnspan=2,padx=5, pady=10, sticky="nsew")

        scrollbar = tk.Scrollbar(self.text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=widget_font)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.output_text.yview)

        self.output_text.config(state=tk.DISABLED)

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        input_label = tk.Label(self.root, text='Input: You can type "quit," "exit," or "q" to end the conversation.', font=label_font)
        input_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.input_text = tk.Text(self.root, font=widget_font, height=4)
        self.input_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
        submit_button.grid(row=4, column=1, padx=5, pady=5)  

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)

        self.html_frame = tk.Frame(self.root)
        self.html_frame.grid(row=1, column=2, rowspan =3, padx=5, pady=5, sticky="nsew")  

        self.html_viewer = HTMLScrolledText(self.html_frame, width=60, height=30)  
        self.html_viewer.pack(fill=tk.BOTH, expand=True)

        self.text_frame.config(width=self.html_viewer.winfo_width())

        

    def display_html(self, file_path):
        """
        Load and display HTML content from a file.
        """
        with open(file_path, "r") as file:
            html_content = file.read()

        self.html_viewer.set_html(html_content)

    def on_submit(self):
        
        input_str = self.input_text.get("1.0", tk.END).strip() 

        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"[AI] Your input is : {input_str}\n")
            
        self.output_text.insert(tk.END, "[AI] Processing your input ...\n")
        self.output_text.config(state=tk.DISABLED)
        self.output_text.see(tk.END)
      
        if not input_str:
            return  

        
        
        input_str = input_str.replace("\n", " ")
        
        self.user_input.set(input_str)
        
        self.input_text.delete("1.0", tk.END)  


    def on_close(self):
        
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.user_input.set('q')

    def tk_print(self, input_str):
        try:
        
            input_str = str(input_str)
        except (ValueError, TypeError):
        
            return

        self.conversation_buffer.append(str(input_str))

    
    def find_yes_no_prompt(self, user_input):
        
        str = """
        Please determine whether the user's input indicates a 'yes' or 'no.' 
        You are an assistant tasked with categorizing user input based on the following three rules. Your goal is to determine which category the input belongs to:

        1. If the user's input indicates a 'yes', in this case, your output is [1]. Example: "Yes." or "Correct"

        2. If the user's input indicates a 'no',n this case, your output is [2]. Example: "No." "I don’t think so." or "Negative"

        3. If the input contains anything else, or a combination of the categories listed above, your output is [3]. Example: "I am a cat."

        Your output should be a single class number, enclosed in square brackets, such as [1], [2] or [3]. Always start and end your output with square brackets.

        Input:
        user_input = "{user_input}"
        """
        prompt= ChatPromptTemplate.from_template(str)
        chain = prompt | self.model   
        input_dict = {
            "user_input":  user_input
        }
        output = chain.invoke(
            input_dict
        )
        self.tk_print(output)
        return output

    def find_best_match(self, user_input, word_list):
        
        
        for word in word_list:
            if word.strip() == user_input.strip():
                self.tk_print("[AI] There is a exact match. {}".format(word.strip()))
                return  word.strip() 

        self.tk_print(f'[AI] There is no exact match for "{user_input}". Looking for the most similar one.')
        
        
        input_word = user_input.lower()

        best_match = ""
        highest_similarity = -1

        for word in word_list:

            word = word.strip()
            tmp_word = word.lower()
            tmp_word = tmp_word.strip()

            distance = Levenshtein.distance(input_word, tmp_word)

            max_len = max(len(input_word), len(word))
            similarity = 1 - (distance / max_len)

            if similarity > highest_similarity:
                best_match = word
                highest_similarity = similarity

        if highest_similarity > 0.3 :
            return best_match
        return ""

        
    def extract_relationship_prompt(self, messages):
        pt_str = """
        You are a smart research assistant. Given input sentences, follow these steps to extract relationships and their connections. 
        The output should be in JSON format only with no explanation.

        ### Step 1: Convert Relationships to Tuples
        Extract the relationships between variables, comparison operators, and values (which can be a single value, a set, or a range defined by the terms "from" and "to"). 
        The variables and values are exactly as they appear in the input. They should not be altered, even if they contain repetitions, phrases, or other variations. 
        The comparison operators are
        "=="   (Is)
        "!="  (Is Not)
        ">"    (Greater than)
        "<"    (Less than)
        ">="  (Greater than or equal to)
        "<="   (Less than or equal to)
        "range" (Define a range)
        "in"  (Define a set of values)
        "not in"  (Exclude a set of values)
        
        - For "is not", the operator is != and the value is after the last "not".
        Example: "SOX9_mutation is not safe" to (SOX9_mutation, deleterious, !=).

        - For "is less than or equal to", the operator is <= and the value must be a number.
        Example: "BMI is less than or equal to 20" to (BMI, 20, <=).
        Example: "TP53_mutation_status is less than or equal to 1" to (TP53_mutation_status, 1, <=)
        
        - For "is greater than or equal to", the operator is >= and the value must be a number.
        Example: "BMI is greater than or equal to 20" to (BMI, 20, >=).
        Example: "KRAS_mutation_status is greater than or equal  to 0" to (TP53_mutation_status, 0, >)

        - Use only 'from' and 'to' to define a range.
        Example: "Age is from 10 to 20" to (age, |10, 20|, range).
        Example: "Dose is from -20.0 to -10.0" to (Dose, |-20.0, -10.0|, range).
        Example: "Blood Pressure is from -20.0 to 30.0" to (Blood Pressure, |-20.0, 30.0|, range).

        - A range always starts and ends with a pipe '|'  

        - Use 'in' or 'not in' to define a set of values. Keep the values as the same as they appear in the input
        Example: "The humidity is in the set |50, 60, 70|" to (humidity, |50, 60, 70|, in).
        Example: "Stage is not in the set |50, 60, 70|" to (humidity, |50, 60, 70|, not in).
        Example: "Tumor stage is in the set |stage I, stage II and IV|." to (age, |stage I, stage II, IV|, in)

        - A set always starts and ends with a pipe '|'. Do not alter anything and keep the set as the same as they appear in the input. 
        Example: "Age is |20, 30 or 40|" to (age, |20, 30, 40|, in).
        Example: "Age is |20a, 30bc and 40df|" to (age, |20a, 30bc, 40df|, in).
        Example: "Age is not 20, 30 and 40" to (age, |20, 30, 40|, not in).
        Example: "Tumor stage is |stage I, II, IV|." to (age, |stage I, II, IV|, in)

        - If no range or set is specified, the input is considered a single value.
        Example: "Age is 30" to (age, 30, ==).
        Example: "Age is not 30" to (age, 30, !=).
        Example: "Age is greater than 30" to (age, 30, >).
        Example: "Age is greater than or equal to 30" to (age, 30, >=).
        Example: "Age is less than 30" to (age, 30, <).
        Example: "Age is less than or equal to 30" to (age, 30, >=).


        - Do not change the value in the input. Allow special characters such as "*" and "!"  in the value.
        Example: "APC_mutation_Amino_Acid_Change is p.S1392*" to (APC_mutation_Amino_Acid_Change, p.S1392*, ==).
        Example: "APC_mutation_Amino_Acid_Change is not p.K19sdfs*9" to (APC_mutation_Amino_Acid_Change, p.K19sdfs*9, !=).

        - Each relationship should be represented as a tuple in the format (variable, value, comparison operator). 
        - Tuples should always start and end with parentheses ( ).

        ### Step 2: Determine the Conjunction
        If multiple relationships (tuples) are connected by "and" or "or," specify the conjunction in the output.  
        If the sentence contains only one tuple, leave the conjunction as an empty string ("").  
        If no conjunction is explicitly mentioned, assume "and" by default.

        ### Step 3: Output as JSON Object
        The output should be a JSON object only with two fields:
        1. "tuples": A list starts with "[" and ends with "]". It contains all the extracted tuples, where each tuple starts and ends with parentheses ( ), and the values are formatted according to the rules above.
        2. "conjunction": A string that indicates whether the relationships are connected by "and," "or," or an empty string for single tuples.

        Input:
        {user_input}

        Output the JSON object only. No explanation or other text.
        If you cannot parse the logic expression, return an empty string.
        Do not generate any program code.

        
        """

        prompt_template = ChatPromptTemplate.from_template(
        pt_str
        )


        chain = prompt_template | llm  
        input_dict = {
                "user_input":messages
            }
        output = chain.invoke(input_dict)

        return output
    
    def replace_bottom_level_conditions(self,expression):
    
        letters =  iter(
            list(string.ascii_uppercase) + 
            list(string.ascii_lowercase) + 
            [u + u for u in string.ascii_uppercase] + 
            [l + l for l in string.ascii_lowercase]
        )   
        condition_dict = {}


        pattern = r'\([^()]+\)'
    
        matches = re.findall(pattern, expression)
    
        for match in matches:
            condition = match.strip()
        
            letter = next(letters)
        
            condition_dict[letter] = condition
        
            expression = expression.replace(condition, " "+letter+" ", 1)
    
        
        return expression, condition_dict
    def check_missing_operator(self, expr):
        expr = expr.replace("(", ' ')
        expr = expr.replace(")", ' ')
        tokens = expr.split()
        token_types = []
        for token in tokens:
            if token == 'and' or token == 'or':
                token_types.append('operator')
            else:
                token_types.append('operand')
        for i in range(1, len(token_types)):
            if token_types[i] == token_types[i - 1]:
                if token_types[i] == 'operand':
                    self.tk_print(f"Missing operator between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False   
                elif token_types[i] == 'operator':
                    self.tk_print(f"Missing operand between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False   

        self.tk_print("No missing operator detected")
        return True

    def has_valid_operators(self, expression):

        expression = expression.replace(" ", "")
    
        
        tokens = re.findall(r'[A-Z]|\(|\)|and|or', expression)
    

        operators = {"and", "or"}
    
        prev_token = None
    
        for i, token in enumerate(tokens):
            if token in operators:

                if prev_token is None or prev_token in operators or prev_token == '(':
                    return False  
            

                if i == len(tokens) - 1:
                    return False  
                next_token = tokens[i + 1]
                if next_token in operators or next_token == ')':
                    return False  
        
            prev_token = token
    
        return True

    def parse_query_I_prompt(self, messages):
        pt_str ="""
        You are an assistant helping to verify users' intentions.
        You will first classify the input sentences into one of the following 4 categories:

        1. ***Explore Data*** The user is asking to explore the dataset. In this case, the system will provide all available attributes and their values.
           For example: Include all samples. Show all data attributes. 
        2. ***Set Up Criteria*** The user is defining a criterion to filter or subset the case cohort.
           For example: Define Parameters to refine the case cohort.Establish Conditions to filter and refine the case cohort.
        3. ***Proceed*** The user is asking to move on and go to the next step. All samples in the selected dataset will be included in your case cohort.
           For example: Let's move on.
        4. ***The Other*** For everything else.
            For example: I am a cat.
        Output the class number, 1, 2, 3 or 4. 
        Your output should always start and end with square brackets [ ].
        
        Input:
        user_input ="{user_input}"

        """

        prompt= ChatPromptTemplate.from_template(pt_str)
        chain = prompt | self.model   
        input_dict = {
            "user_input":  messages
        }
        output = chain.invoke(
            input_dict
        )

        return output
    
    def run_script(self, script_fname, arg_fname ):
        command = ['python3', script_fname, arg_fname]
    
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Script executed successfully.")
            print(result.stdout)  
        except subprocess.CalledProcessError as e:
            print("Error while running script.")
            print(e.stderr)

         

    def parse_query_II_prompt(self, messages):
        pt_str ="""
        You are an assistant tasked with categorizing user input based on the following three rules. Your goal is to determine which category the input belongs to:

        1. If the input mentions anything related to an "Odds Ratio Test", the user is requesting an odds ratio test based on a clinical context defined by the user. In this case, your output is [1]. Example: "Perform an odds ratio test for patients."

        2. If the input refers to "Survival Analysis", the user is requesting survival analysis for the case and control cohorts. In this case, your output is [2]. Example: "Run survival analysis on patients with high gene expression."

        3. If the input contains anything else, or a combination of the categories listed above, your output is [3]. Example: "I am a cat."

        Your output should be a single class number, enclosed in square brackets, such as [1], [2] or [3]. Always start and end your output with square brackets.

        Input:
        user_input = "{user_input}"

        """

        prompt= ChatPromptTemplate.from_template(pt_str)
        chain = prompt | self.model   
        input_dict = {
            "user_input":  messages
        }
        output = chain.invoke(
            input_dict
        )
       
        return output
    


    def return_rownames(self, metadata_df, attr, opr, value_str):
        attr = attr.replace('"', "")
        opr = opr.replace('"', "")
        value_str = value_str.replace('"', "")

        sample_list =[]
        attribute_id  = self.find_best_match( attr, metadata_df.columns)
        if attribute_id =="":
            self.tk_print("[AI] I can't find the attribute {} in the metadata.".format(attr))
            return None
        
        self.tk_print(f"[AI] {attribute_id} is used here.")
        value_str = value_str.strip()
        value_list = []
        if value_str.startswith("|") and value_str.endswith("|"):
            value_list = [x.strip() for x in value_str.strip('|').split(',')]
        else:
            value_list.append(value_str)
        
        for value in value_list:
            if ptypes.is_numeric_dtype(metadata_df[attribute_id]):
                max_value = metadata_df[attribute_id].max()
                min_value = metadata_df[attribute_id].min()

                try:
                    value_d = float(value)
                except ValueError:
                    self.tk_print(f"[AI] Error: The string '{value}' cannot be converted to a float.")
                    return None
        
            else:
                unique_values = metadata_df[attribute_id].unique()
                valid_list = []
                for item in unique_values:
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() not in valid_list:
                                    valid_list.append(substring.strip())
                        else:
                            if item not in valid_list:
                                valid_list.append(item.strip())

                if not (value in valid_list):
                    self.tk_print(f'[AI] {value} is not in [ {", ".join(valid_list)} ]. Use "explore data" to check valid values for the data attributes.')
                    return None
        opr = opr.strip() 
        if ptypes.is_numeric_dtype(metadata_df[attribute_id]):
            if opr == ">" :
                sample_list = metadata_df.index[metadata_df[attribute_id] > float(value_list[0])]
            if opr == ">=" :
                sample_list = metadata_df.index[metadata_df[attribute_id] >= float(value_list[0])]
            if opr == "<" :
                sample_list = metadata_df.index[metadata_df[attribute_id] < float(value_list[0])]
            if opr == "<=" :
                sample_list = metadata_df.index[metadata_df[attribute_id] <= float(value_list[0])]
            if opr == "==" or opr == "in":
                value_d_list = [float(x) for x in value_list]
                matching_row_names = metadata_df[metadata_df[attribute_id].astype(float).isin(value_d_list)].index
                sample_list = matching_row_names.tolist()
            if opr == "!=" or opr == "not in":
                value_d_list = [float(x) for x in value_list]
                matching_row_names = metadata_df[~metadata_df[attribute_id].astype(float).isin(value_d_list)].index
                sample_list = matching_row_names.tolist()
                sample_set = set(sample_list)
                rows_not_in_sample_list = set(metadata_df.index) - sample_set
                sample_list = list(rows_not_in_sample_list)

            if opr == "range":
                value_d_list = [float(x) for x in value_list]
                min_value = min(value_d_list)
                max_value = max(value_d_list)
                matching_row_names =  metadata_df[(metadata_df[attribute_id].astype(float) >= min_value) & (metadata_df[attribute_id].astype(float) <= max_value)].index
                sample_list = matching_row_names.tolist()
        else:
            if opr == ">" or opr == "<" or opr == ">=" or opr == "<=" or opr == "range":
                self.tk_print(f'[AI] The non-numeric values is not comparable using {opr}.')
                return None
            elif opr == "==" or opr == "in":
                sample_list=[]
                for index, item in zip(metadata_df.index, metadata_df[attribute_id]):
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                sample_list.append(index)
            elif opr == "!=" or opr == "not in":
                sample_list=[]
                for index, item in zip(metadata_df.index, metadata_df[attribute_id]):
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                sample_list.append(index)
                sample_set = set(sample_list)
                rows_not_in_sample_list = set(metadata_df.index) - sample_set
                sample_list = list(rows_not_in_sample_list)

        return sample_list

    def infix_to_postfix(self, expression):
        output = []
        operator_stack = []
    
        tokens = expression.split()

        for token in tokens:
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()   
            elif token in {'and', 'or'}:
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                output.append(token)

        while operator_stack:
            output.append(operator_stack.pop())

        return output

    def evaluate_postfix(self, expression, sample_dict):
        stack = []
        i=0
        for token in expression:
            token= token.strip()

            if token in sample_dict.keys():  
                stack.append(token)
            elif token == 'or': 
                operand2 = stack.pop()
                operand1 = stack.pop()
                list1 = sample_dict[operand1]
                list2 = sample_dict[operand2]
                union_list = list(set(list1) | set(list2))
                sample_dict["$"+str(i)] = union_list
                stack.append("$"+str(i))
                i=i+1
            elif token == 'and': 
                operand2 = stack.pop()
                operand1 = stack.pop()
                list1 = sample_dict[operand1]
                list2 = sample_dict[operand2]
                intersection_list = list(set(list1) & set(list2))
                sample_dict["$"+str(i)] = intersection_list
                stack.append("$"+str(i))
                i=i+1
        
        return sample_dict[stack.pop()]


    def check_balanced_parentheses(self, input_string):
        stack = []

        for char in input_string:
            if char == '(':
                stack.append('(')
            elif char == ')':
                if len(stack) == 0:
                    return False
                stack.pop()

        if len(stack) == 0:
            return True
        else:
            return False
    
    def init_Case_fun(self, state: AgentState):
        
        with open('dialogs/init_case_1.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

        self.tk_print("The following datasets have been successfully installed in your computer:\n")
        df = pd.read_csv('data/dataset.tsv', sep='\t' ,na_values=["none", ""])
        first_col_width = 30
        second_col_width = 50
        df['Name'] = df['Name'].str.slice(0, first_col_width)
        df['Description'] = df['Description'].str.slice(0, second_col_width)

        header = f"{'Name'.ljust(first_col_width)}      {'Description'.ljust(second_col_width)}"
        self.tk_print(header)

        self.tk_print("-" * ( first_col_width + second_col_width ))

        for _, row in df.iterrows():
            formatted_row = f"{row['Name'].ljust(first_col_width)}{row['Description'].rjust(second_col_width)}"
            self.tk_print(formatted_row)

        self.data_repository = df['Name'].to_list()
        str="\n[AI] What is the dataset you want to use for the case samples? Please input the name of the dataset.\n"
        self.tk_print(str)
        pass
        

        
    def load_data_Case_fun(self, state: AgentState):
        index_fname = "data/{}/INDEX.tsv".format(self.Case_data_id )
        df = pd.read_csv(index_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])

        config_dict = {}
        
        for key in df.index:
            config_dict[key] = df.loc[ key, "value" ] 
        
        metadata_fname = "data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] )
        self.Case_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
        self.Case_config_dict = config_dict
        self.Case_metadata_df = df
       
        self.Case_metadata_df = self.Case_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
       
        rows, columns = self.Case_metadata_df.shape
        str=f"[AI] Your data table is located at {metadata_fname}.\n"
        self.tk_print(str)
        str=f"[AI] There are {rows} samples and {columns} attributes in your dataset.\n"
        self.tk_print(str)


    def input_data_Case_fun(self, state: AgentState):

        messages = state['messages'][-1].content
        
        output = self.find_best_match(messages, self.data_repository  )
        messages = "2"
        
        if  output==""  :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        else :
                self.Case_data_id = output
                self.tk_print("\n[AI] {} is used here.\n".format(self.Case_data_id))
                
                if self.Case_data_id in self.data_repository :
                    messages = "1"
                else:
                    self.tk_print("\n[AI] ***WARNING*** Your input is invalid. Please try again.\n")

        return {'messages': [messages]}

        
    def init_query_I_Case_fun(self, state: AgentState):

        with open('dialogs/parse_query_I.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="\n=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        
        if len(self.Case_criteria_logic) ==0:
            str ="[AI] You have not defined any criteria to filter samples in the selected dataset for the case cohort.\n"
            self.tk_print(str)
        else:
            str =f"[AI] You have defined {len(self.Case_criteria_logic)} criteria and selected {len(self.Case_sample_ids)} samples for the case cohort."
            self.tk_print(str)

        
        str = loaded_dict["message"]
        self.tk_print(str)
        
        
        str ="[AI] What would you like to do next? \n"
        self.tk_print(str)

    def init_query_I_Ctrl_fun(self, state: AgentState):

        with open('dialogs/parse_query_I.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="\n=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)

        if len(self.Ctrl_criteria_logic) ==0:
            str ="[AI] You have not defined any criteria to filter samples in the selected dataset for the case cohort.\n"
            self.tk_print(str)
        else:
            str =f"[AI] You have defined {len(self.Ctrl_criteria_logic)} criteria and selected {len(self.Ctrl_sample_ids)} samples for the case cohort."
            self.tk_print(str)
        

        str = loaded_dict["message"]
        self.tk_print(str)

        
        str ="[AI] What would you like to do next? \n"
        self.tk_print(str)

    def parse_query_I_fun(self, state: AgentState):
        
        messages = state['messages'][-1].content
        
        output = self.parse_query_I_prompt(messages)
        
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "4"
        
        for match in matches: 
                data_id_list =  match.split(',')
                if data_id_list != []:
                    for data_id in data_id_list:
                        output = data_id.strip()

        if data_id_list == [] :
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

        if data_id_list != []:
    
            if len(matches) > 1 or len(data_id_list)>1 :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
            else :
                if output == "1":
                    self.tk_print("\n[AI] You want to explore data.\n" )

                elif output == "2":
                    self.tk_print("\n[AI] You want to set up criteria.\n" )

                elif output == "3":
                    self.tk_print("\n[AI] You want to proceed.\n" )
                else:
                    self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                # messages = "Yes"

        return {'messages': [output]}
        
        
     
    def init_set_criteria_fun(self, state: AgentState):
        with open('dialogs/set_up_criteria.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        
        pass
    def make_decision_fun(self, state: AgentState):
        messages = state['messages'][-1].content

        return int(messages)
        
    def overview_Case_fun(self, state: AgentState): 
        str ="\n=======================================================\n"+ "Introduction to the Case Dataset" +"\n=======================================================\n"
        self.tk_print(str)
        index_fname = "data/{}/{}".format(self.Case_data_id, self.Case_config_dict["README"])
        with open(index_fname, "r") as f:
            file_content = f.read()
        f.close()
        self.tk_print(file_content)

        self.tk_print("[AI] Please enter the name of a data attribute, and I can display the distribution of its values.")
   

    def show_attr_values_Case_fun(self, state: AgentState):

        self.tk_print("show_attr_values_Case_fun")
        messages = state['messages'][-1].content
        data_attr = self.find_best_match(messages, self.Case_metadata_df.columns  )
        if data_attr != "" :
            self.tk_print(data_attr)

            msg_dict ={
            "metafname":self.Case_metafname,    
            "Attr_ID":data_attr,
            "output_path":self.conversation_path,
            "output_png":self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".png",
            "output_html":self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".html",
            "output_pdf":self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".pdf"
            }
        
            with open( self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".pkl", 'wb') as f:
                pickle.dump(msg_dict, f)
            f.close()
            self.run_script( "EXHIBIT_Agent.py",self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".pkl" )
            self.html_fname = self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".html"

            self.case_exhibit_num = self.case_exhibit_num+1
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

    def set_criteria_Case_fun(self, state: AgentState):
        sample_dict = {}
        output = ""
        messages = state['messages'][-1].content
        messages = messages.replace("{", '|')
        messages = messages.replace("}", '|')
        messages = "("+messages+")"
        if '(' in messages or ')' in messages:
            if not self.check_balanced_parentheses(messages) :
                self.tk_print("[AI] parentheses are not closed")
                return {'messages': [output]}

       
        new_expression, condition_map = self.replace_bottom_level_conditions(messages)
        
        cleaned_expression = re.sub(r'[A-Z\s()]+|and|or', '', new_expression)
        if cleaned_expression != '':
            self.tk_print("[AI] The input is not a valid expression.")
            return {'messages': [output]}
            
        if self.check_missing_operator(new_expression) == False:
            self.tk_print("[AI] there are missing operators.")
            return {'messages': [output]}

        if self.has_valid_operators(new_expression) == False:
            self.tk_print("[AI] The input is not a valid expression.")
            return {'messages': [output]}
        
        postorder_list = self.infix_to_postfix(new_expression)
        for letter, condition in condition_map.items():

            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)
            
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            tuple_match = re.search(list_pattern, input_string, re.DOTALL)
            conjunction_match = re.search(conjunction_pattern, input_string)

            if tuple_match is None or conjunction_match is None:
                self.tk_print("[AI] Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
               
                tuples_str = tuple_match.group(1).strip()
                tuple_pattern = r'\(([^)]+)\)'
                matches = re.findall(tuple_pattern, input_string, re.DOTALL)
                if matches is None:
                    self.tk_print("[AI] Cannot parse the logic expression!")
                    return {'messages': [output]} 
                else:
                    if len(matches) >1:
                        self.tk_print("[AI] There are more than 1 relationship defined in a sentence.")
                        return {'messages': [output]} 
                    else:
                        for tuple_str in matches:
                            token_list = tuple_str.split(",")
                            attr = token_list[0]
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            self.tk_print(f'[AI] I think it means "{attr} {opr} {middle_words}".' )
                            sample_list = self.return_rownames(self.Case_metadata_df, attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No sample matches this criteria. "+tuple_str)

                                return {'messages': [output]} 
                            else:
                                sample_dict[letter] = sample_list
                                

        sample_list = self.evaluate_postfix(postorder_list, sample_dict )
        
        self.Case_sample_ids = sample_list
        out_html_fname = self.conversation_path+"/case_sample_selection.html"
        self.Case_criteria_str = new_expression
        self.Case_criteria_logic = condition_map
        msg_dict ={
        "case_id":"Case",
        "total_num":self.Case_metadata_df.shape[0],
        "criteria_str":self.Case_criteria_str ,
        "criteria_logic":self.Case_criteria_logic ,
        "selected_num":len(self.Case_sample_ids),
        "output_path":self.conversation_path,
        "output_png":self.conversation_path+"/case_sample_selection_"+str(self.case_DS_num)+".png",
        "output_html":self.conversation_path+"/case_sample_selection_"+str(self.case_DS_num)+".html",
        "output_pdf":self.conversation_path+"/case_sample_selection_"+str(self.case_DS_num)+".pdf"
        }

        with open( self.conversation_path+'/case_sample_selection+'+str(self.case_DS_num)+'.pkl', 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        
        time.sleep(1)
        self.run_script( "DS_Agent.py",self.conversation_path+'/case_sample_selection+'+str(self.case_DS_num)+'.pkl' )
        self.html_fname = msg_dict["output_html"]
       
        
        self.tk_print(f"[AI] Congratulations! You have successfully set up the criteria to refine the samples. You can now proceed to the next step.\n" )
        self.case_DS_num = self.case_DS_num+1
        return {'messages': [output]}  

    
        
    def summary_Case_fun(self, state: AgentState): 
        self.html_fname = "dialogs/welcome_2.html"
        if len(self.Case_criteria_logic) ==0:
            self.Case_sample_ids = self.Case_metadata_df.index.to_list()
            str =f"[AI] You have not defined any criteria to filter samples for the case cohort. All {len(self.Case_sample_ids)} samples in the dataset will be included."
            self.tk_print(str)
           
        else:
            str =f"[AI] You have defined {len(self.Case_criteria_logic)} criteria and selected {len(self.Case_sample_ids)} samples for the case cohort."
            self.tk_print(str)

        
         
        

    def init_Ctrl_fun(self, state: AgentState):
        with open('dialogs/init_ctrl_1.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        
        
        
        self.tk_print("The following datasets have been successfully installed in your computer:\n")
        df = pd.read_csv('data/dataset.tsv', sep='\t' ,na_values=["none", ""])
        first_col_width = 30
        second_col_width = 50
        df['Name'] = df['Name'].str.slice(0, first_col_width)
        df['Description'] = df['Description'].str.slice(0, second_col_width)


        header = f"{'Name'.ljust(first_col_width)}      {'Description'.ljust(second_col_width)}"
        self.tk_print(header)

        self.tk_print("-" * ( first_col_width + second_col_width ))

        for _, row in df.iterrows():
            formatted_row = f"{row['Name'].ljust(first_col_width)}{row['Description'].rjust(second_col_width)}"
            self.tk_print(formatted_row)

        self.data_repository = df['Name'].to_list()
        str="\n[AI] What is the dataset you want to use for the control samples? Please input the name of the dataset.\n"
        self.tk_print(str)
        pass
        
    
    def input_data_Ctrl_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        
        output = self.find_best_match(messages, self.data_repository  )
        messages = "2"
       
        if  output==""  :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        else :
                self.Ctrl_data_id = output
                self.tk_print("\n[AI] {} is used here.\n".format(self.Ctrl_data_id))

                if self.Ctrl_data_id in self.data_repository :
                    messages = "1"
                else:
                    self.tk_print("\n[AI] ***WARNING*** Your input is invalid. Please try again.\n")

        return {'messages': [messages]}

    def load_data_Ctrl_fun(self, state: AgentState):
        index_fname = "data/{}/INDEX.tsv".format(self.Ctrl_data_id )
        df = pd.read_csv(index_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])

        config_dict = {}
        
        for key in df.index:
            config_dict[key] = df.loc[ key,  "value"  ]  
        metadata_fname = "data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] )
        self.Ctrl_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
    
        self.Ctrl_metadata_df = df
        self.Ctrl_config_dict = config_dict
        self.Ctrl_metadata_df = self.Ctrl_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        rows, columns = self.Ctrl_metadata_df.shape
        
        str=f"[AI] Your data table is located at {metadata_fname}.\n"
        self.tk_print(str)
        str=f"[AI] There are {rows} samples and {columns} attributes in your dataset.\n"
        self.tk_print(str)



        
    def overview_Ctrl_fun(self, state: AgentState): 
        str ="\n=======================================================\n"+ "Introduction to the Control Dataset" +"\n=======================================================\n"
        self.tk_print(str)
        index_fname = "data/{}/{}".format(self.Ctrl_data_id, self.Ctrl_config_dict["README"])
        with open(index_fname, "r") as f:
            file_content = f.read()
        f.close()
        self.tk_print(file_content)

        self.tk_print("[AI] Please enter the name of a data attribute, and I can display the distribution of its values.")
    
    def show_attr_values_Ctrl_fun(self, state: AgentState):
        self.tk_print("show_attr_values_Ctrl_fun")
        messages = state['messages'][-1].content
        data_attr = self.find_best_match(messages, self.Ctrl_metadata_df.columns  )
        if data_attr != "" :
            self.tk_print(data_attr)

            msg_dict ={
            "metafname":self.Ctrl_metafname,    
            "Attr_ID":data_attr,
            "output_path":self.conversation_path,
            "output_png":self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".png",
            "output_html":self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".html",
            "output_pdf":self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".pdf"
            }
        
            with open( self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".pkl", 'wb') as f:
                pickle.dump(msg_dict, f)
            f.close()
            self.run_script( "EXHIBIT_Agent.py",self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".pkl" )
            self.html_fname = self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".html"

            self.ctrl_exhibit_num = self.ctrl_exhibit_num+1
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

    def set_criteria_Ctrl_fun(self, state: AgentState):
    
        sample_dict = {}
        output = ""
        messages = state['messages'][-1].content
        messages = messages.replace("{", '|')
        messages = messages.replace("}", '|')
        messages = "("+messages+")"
        if '(' in messages or ')' in messages:
            if not self.check_balanced_parentheses(messages) :
                self.tk_print("[AI] parentheses are not closed")
                return {'messages': [output]}

       
        new_expression, condition_map = self.replace_bottom_level_conditions(messages)
        

        cleaned_expression = re.sub(r'[A-Z\s()]+|and|or', '', new_expression)
        if cleaned_expression != '':
            self.tk_print("[AI] The input is not a valid expression.")
            return {'messages': [output]}
            
        if self.check_missing_operator(new_expression) == False:
            self.tk_print("[AI] there are missing operators.")
            return {'messages': [output]}

        if self.has_valid_operators(new_expression) == False:
            self.tk_print("[AI] The input is not a valid expression.")
            return {'messages': [output]}
        
        postorder_list = self.infix_to_postfix(new_expression)

        for letter, condition in condition_map.items():

            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)

            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            conjunction_match = re.search(conjunction_pattern, input_string)


            if tuple_match is None or conjunction_match is None:
                self.tk_print("[AI] Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
               
                tuples_str = tuple_match.group(1).strip()

                tuple_pattern = r'\(([^)]+)\)'
                matches = re.findall(tuple_pattern, input_string, re.DOTALL)
                if matches is None:
                    self.tk_print("[AI] Cannot parse the logic expression!")
                    return {'messages': [output]} 
                else:
                    if len(matches) >1:
                        self.tk_print("[AI] There are more than 1 relationship defined in a sentence.")
                        return {'messages': [output]} 
                    else:
                        for tuple_str in matches:
                            token_list = tuple_str.split(",")
                            attr = token_list[0]
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            self.tk_print(f'[AI] I think it means "{attr} {opr} {middle_words}".' )
                            sample_list = self.return_rownames(self.Ctrl_metadata_df, attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No sample matches this criteria. "+tuple_str)

                                return {'messages': [output]} 
                            else:
                                sample_dict[letter] = sample_list
                                

        sample_list = self.evaluate_postfix(postorder_list, sample_dict )
        
        self.Ctrl_sample_ids = sample_list
        out_html_fname = self.conversation_path+"/ctrl_sample_selection.html"
        self.Ctrl_criteria_str = new_expression
        self.Ctrl_criteria_logic = condition_map
        msg_dict ={
        "case_id":"Control",
        "total_num":self.Ctrl_metadata_df.shape[0],
        "criteria_str":self.Ctrl_criteria_str ,
        "criteria_logic":self.Ctrl_criteria_logic ,
        "selected_num":len(self.Ctrl_sample_ids),
        "output_path":self.conversation_path,
        "output_png":self.conversation_path+"/ctrl_sample_selection_"+str(self.ctrl_DS_num)+".png",
        "output_html":self.conversation_path+"/ctrl_sample_selection_"+str(self.ctrl_DS_num)+".html",
        "output_pdf":self.conversation_path+"/ctrl_sample_selection_"+str(self.ctrl_DS_num)+".pdf"
        }

        with open( self.conversation_path+'/ctrl_sample_selection+'+str(self.ctrl_DS_num)+'.pkl', 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()

      
        time.sleep(1)
        self.run_script( "DS_Agent.py",self.conversation_path+'/ctrl_sample_selection+'+str(self.ctrl_DS_num)+'.pkl' )
        self.html_fname = msg_dict["output_html"]
        self.ctrl_DS_num = self.ctrl_DS_num+1
        self.tk_print(f"[AI] Congratulations! You have successfully set up the criteria to refine the control samples. You can now proceed to the next step.\n" )
        
        return {'messages': [output]}  
  

    
    def summary_Ctrl_fun(self, state: AgentState): 
        self.html_fname = "dialogs/welcome_3.html"
      
        if len(self.Case_criteria_logic) ==0:
            self.Ctrl_sample_ids = self.Ctrl_metadata_df.index.to_list()
            str =f"[AI] You have not defined any criteria to filter samples for the control cohort. All {len(self.Ctrl_sample_ids)} samples in the dataset will be included."
            self.tk_print(str)
        else:
            str =f"[AI] You have defined {len(self.Ctrl_criteria_logic)} criteria and selected {len(self.Ctrl_sample_ids)} samples for the case cohort."
            self.tk_print(str)
        output = "1"
        
        if self.Ctrl_data_id == self.Ctrl_data_id :
            if set(self.Ctrl_sample_ids) & set(self.Case_sample_ids) :
                
                str =f"[AI] *** Warning *** There are {len(set(self.Ctrl_sample_ids) & set(self.Case_sample_ids))} samples shared in the case and control cohorts. Please revise the sample selection."
                self.tk_print(str)
                output = "2"



        
        return {'messages': [output]}


    
    def init_exec_fun(self, state: AgentState):
        with open('dialogs/init_exec.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)    
        
    def parse_exec_fun(self, state: AgentState):    
        
        messages = state['messages'][-1].content
        
        output = self.parse_query_II_prompt(messages)
        
        matches = re.findall(r'\[(.*?)\]', output)
        data_id_list = []
        output = "3"
        
        for match in matches: 
                data_id_list =  match.split(',')
                if data_id_list != []:
                    for data_id in data_id_list:
                        output = data_id.strip()

        if data_id_list == [] :
            tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
            output = "3"
            return {'messages': [output]}

        if data_id_list != []:
            
    
            if len(matches) > 1 or len(data_id_list)>1 :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                output = "3"
                return {'messages': [output]}
            else :
                if output == "1":
                    self.tk_print("\n[AI] You want to test the odds ratio based on clinical conditions\n" )
                    output = "1"
                    return {'messages': [output]}
                elif output == "2":
                    self.tk_print("\n[AI] You want to conduct survival analysis based on the cohort data.\n" )
                    output = "2"
                    return {'messages': [output]}
                else:
                    self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                    output = "3"
                    return {'messages': [output]}

        return {'messages': [output]}

    def init_OR_fun(self, state: AgentState):   
        with open('dialogs/init_OR.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)    
        
    def parse_OR_fun(self, state: AgentState): 
        Case_sample_dict = {}
        Ctrl_sample_dict = {}
        output = ""
        messages = state['messages'][-1].content
        messages = messages.replace("{", '|')
        messages = messages.replace("}", '|')
        messages = "("+messages+")"
        if '(' in messages or ')' in messages:
            if not self.check_balanced_parentheses(messages) :
                self.tk_print("not closed")
                return {'messages': [output]}

     
        new_expression, condition_map = self.replace_bottom_level_conditions(messages)
        
    

        cleaned_expression = re.sub(r'[A-Z\s()]+|and|or', '', new_expression)
        if cleaned_expression != '':
            self.tk_print("[AI] Your input is not a valid expression.")
            return {'messages': [output]}
            
        if self.check_missing_operator(new_expression) == False:
            self.tk_print("[AI] There are missing operators.")
            return {'messages': [output]}

        if self.has_valid_operators(new_expression) == False:
            self.tk_print("[AI] Operators are not valid")
            return {'messages': [output]}
        
        postorder_list = self.infix_to_postfix(new_expression)
       
        for letter, condition in condition_map.items():

            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)
           
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            conjunction_match = re.search(conjunction_pattern, input_string)

            self.tk_print(tuple_match)
            self.tk_print(conjunction_match)
            if tuple_match is None or conjunction_match is None:
                self.tk_print("Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
               
                tuples_str = tuple_match.group(1).strip()
                self.tk_print(  tuples_str)

                tuple_pattern = r'\(([^)]+)\)'
                matches = re.findall(tuple_pattern, input_string, re.DOTALL)
                if matches is None:
                    self.tk_print("[AI] Cannot parse the logic expression!")
                    return {'messages': [output]} 
                else:
                    if len(matches) >1:
                        self.tk_print("[AI] There are more than 1 relationship in the sentence.")
                        return {'messages': [output]} 
                    else:
                        for tuple_str in matches:
                            self.tk_print(tuple_str)
                            token_list = tuple_str.split(",")
                            attr = token_list[0]
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            self.tk_print(f'[AI] I think it means "{attr} {opr} {middle_words}".' )
                            sample_list = self.return_rownames(self.Case_metadata_df.loc[self.Case_sample_ids], attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No Case sample matched the criteria.")
                                return {'messages': [output]} 
                            else:
                                Case_sample_dict[letter] = sample_list

                            sample_list = self.return_rownames(self.Ctrl_metadata_df.loc[self.Ctrl_sample_ids], attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No Control sample matched the criteria.")
                                return {'messages': [output]} 
                            else:
                                Ctrl_sample_dict[letter] = sample_list
                            
                                

        Case_sample_list = self.evaluate_postfix(postorder_list, Case_sample_dict )
        self.tk_print(f"[AI] There are {len(Case_sample_list)} matched Case samples." )
        Ctrl_sample_list = self.evaluate_postfix(postorder_list, Ctrl_sample_dict )
        self.tk_print(f"[AI] There are {len(Ctrl_sample_list)} matched Control samples." )


        a = len(Case_sample_list) 
        b = len(self.Case_sample_ids) - len(Case_sample_list)
        c = len(Ctrl_sample_list) 
        d = len(self.Ctrl_sample_ids) - len(Ctrl_sample_list)
        
        msg_dict ={
        "Case_in":a,
        "Case_out":b,
        "Ctrl_in":c,
        "Ctrl_out":d,
        "criteria_str":new_expression ,
        "criteria_logic":condition_map ,
        "output_path":self.conversation_path,
        "output_png":self.conversation_path+"/OR_test_"+str(self.or_num) +".png",
        "output_html":self.conversation_path+"/OR_test_"+str(self.or_num) +".html",
        "output_pdf":self.conversation_path+"/OR_test_"+str(self.or_num) +".pdf"
        }

        with open( self.conversation_path+"/OR_test_"+str(self.or_num) +".pkl", 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        self.run_script( "OR_Agent.py",self.conversation_path+"/OR_test_"+str(self.or_num) +".pkl" )
        self.html_fname = self.conversation_path+"/OR_test_"+str(self.or_num) +".html"

        self.or_num = self.or_num+1
        return {'messages': [output]}  
      

    def init_Survival_fun(self, state: AgentState):
        with open('dialogs/init_Survival.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  
        pass
    def parse_Survival_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output = self.find_yes_no_prompt(messages)
        
        matches = re.findall(r'\[(.*?)\]', output)
        data_id_list = []
        output = "3"
        
        for match in matches: 
                data_id_list =  match.split(',')
                if data_id_list != []:
                    for data_id in data_id_list:
                        output = data_id.strip()

        if data_id_list == [] :
            tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
            output = "5"
            return {'messages': [output]}

        if data_id_list != []:
            
    
            if len(matches) > 1 or len(data_id_list)>1 :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                output = "3"
                return {'messages': [output]}
            else :
                if output == "1":
                    self.tk_print("\n[AI] You want to conduct a multivariate survival analysis.\n" )
                    output = "1"
                    return {'messages': [output]}
                elif output == "2":
                    self.tk_print("\n[AI] You aim to perform a univariate survival analysis using the cohort data.\n" )
                    output = "2"
                    return {'messages': [output]}
                else:
                    self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                    output = "3"
                    return {'messages': [output]}

        return {'messages': [output]}
        
    
    def init_multiple_Survival_fun(self, state: AgentState):
        with open('dialogs/multi_Survival.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  
        pass

    def multiple_Survival_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        tmp_list = messages.split(",")
        for item in tmp_list:
            Case_item = self.find_best_match(item,self.Case_metadata_df.columns)
            Ctrl_item = self.find_best_match(item,self.Ctrl_metadata_df.columns)
            if Case_item =="":
                self.tk_print(f"[AI] Your input {item} is not a valid data attribute name in the case cohort. We will skip it.") 
            if Ctrl_item =="":
                self.tk_print(f"[AI] Your input {item} is not a valid data attribute name in the control cohort. We will skip it.") 
            if Case_item !="" and Ctrl_item !="" and Case_item == Ctrl_item and self.Ctrl_metadata_df[Ctrl_item].dtype == self.Case_metadata_df[Case_item].dtype:
                self.surv_extra.append(Case_item)
        


    def run_Survival_fun(self, state: AgentState):
       
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "Ctrl_metafname":self.Ctrl_metafname,
        "Ctrl_ID":self.Ctrl_sample_ids,
        "output_path":self.conversation_path,
        "output_OS_png":self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png",
        "output_PFS_png":self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png",
        "output_forest_OS_png":self.conversation_path+"/Surv_forest_OS_"+str(self.surv_num) +".png",
        "output_forest_PFS_png":self.conversation_path+"/Surv_forest_PFS_"+str(self.surv_num) +".png",
        "output_html":self.conversation_path+"/Surv_"+str(self.surv_num) +".html",
        "output_pdf":self.conversation_path+"/Surv_"+str(self.surv_num) +".pdf"
        }
        
        OS_flag =1 
        if "OS_TIME" in self.Case_config_dict and self.Case_config_dict["OS_TIME"].strip() !="":
            pass
        else:
            OS_flag=0

        if "OS_TIME" in self.Ctrl_config_dict and self.Ctrl_config_dict["OS_TIME"].strip() !="":
            pass
        else:
            OS_flag=0
        
        if "OS_STATUS" in self.Case_config_dict and self.Case_config_dict["OS_STATUS"].strip() !="":
            pass
        else:
            OS_flag=0

        if "OS_STATUS" in self.Ctrl_config_dict and self.Ctrl_config_dict["OS_STATUS"].strip() !="":
            pass
        else:
            OS_flag=0
        
        if OS_flag ==1:
            msg_dict["output_OS_png"]=self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png"
            msg_dict["Case_OS_TIME"] = self.Case_config_dict["OS_TIME"] 
            msg_dict["Case_OS_STATUS"] = self.Case_config_dict["OS_STATUS"] 
            msg_dict["Ctrl_OS_TIME"] = self.Ctrl_config_dict["OS_TIME"] 
            msg_dict["Ctrl_OS_STATUS"] = self.Ctrl_config_dict["OS_STATUS"] 

        
        PFS_flag =1 
        if "PFS_TIME" in self.Case_config_dict and self.Case_config_dict["PFS_TIME"].strip() !="":
            pass
        else:
            PFS_flag=0

        if "PFS_TIME" in self.Ctrl_config_dict and self.Ctrl_config_dict["PFS_TIME"].strip() !="":
            pass
        else:
            PFS_flag=0
        
        if "PFS_STATUS" in self.Case_config_dict and self.Case_config_dict["PFS_STATUS"].strip() !="":
            pass
        else:
            PFS_flag=0

        if "PFS_STATUS" in self.Ctrl_config_dict and self.Ctrl_config_dict["PFS_STATUS"].strip() !="":
            pass
        else:
            PFS_flag=0
        
        if PFS_flag ==1:
            msg_dict["output_PFS_png"]=self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png"
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 
            msg_dict["Ctrl_PFS_TIME"] = self.Ctrl_config_dict["PFS_TIME"] 
            msg_dict["Ctrl_PFS_STATUS"] = self.Ctrl_config_dict["PFS_STATUS"] 
        

        
        msg_dict["EXTRA_ATTR"] = self.surv_extra
        

        with open( self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl", 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        self.run_script( "SURV_Agent.py",self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" )
        self.html_fname = self.conversation_path+"/Surv_"+str(self.surv_num) +".html"
        self.surv_extra = []
        self.surv_num = self.surv_num+1
        
    def run(self,thread, thread_id):

        self.thread_id = thread_id
        user_input = ""
        current_directory = os.getcwd()
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(current_directory+"/conversations/"+current_time)
     
        self.conversation_path = current_directory+"/conversations/"+current_time
        for event in self.graph.stream({"messages": ("user", user_input)} ,thread):
            for value in event.values():
                pass
        snapshot = self.graph.get_state(thread)
        
        while len(snapshot.next)>0:
            
            conversation_content = "\n".join(self.conversation_buffer)
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, conversation_content+"\n")
            self.output_text.config(state=tk.DISABLED)
            self.output_text.see(tk.END)
            
            self.conversation_buffer=[]
            
            self.display_html(self.html_fname)

            self.user_input.set('')   
            self.root.wait_variable(self.user_input)

            input_str = self.user_input.get()
            
            

            if snapshot.next[0]=="input_data_Case" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_Case"
                )

            if snapshot.next[0]=="input_data_Ctrl" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_Ctrl"
                )

            if snapshot.next[0]=="show_attr_values_Case" :
                
                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "overview_Case"
                )
            if snapshot.next[0]=="show_attr_values_Ctrl" :
             
                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "overview_Ctrl"
                )
            
            if snapshot.next[0]=="set_criteria_Case" :
               
                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_set_criteria_Case"
                )

            if snapshot.next[0]=="set_criteria_Ctrl" :
                
                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_set_criteria_Ctrl"
                )

            if snapshot.next[0]=="parse_query_I_Case" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_query_I_Case"
                )
            
            if snapshot.next[0]=="parse_query_I_Ctrl" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_query_I_Ctrl"
                )
            
            if snapshot.next[0]=="parse_exec" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_exec"
                )
            
            if snapshot.next[0]=="parse_OR" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_OR"
                )
            
            if snapshot.next[0]=="parse_Survival" :

                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_Survival"
                )

            if snapshot.next[0]=="multiple_Survival" :
                print("test")
                if input_str.lower() in ["quit", "exit", "q"]:
                    self.tk_print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_multiple_Survival"
                )
            

            for event in self.graph.stream(None ,thread):
                for value in event.values():
                    pass
            snapshot = self.graph.get_state(thread)
            if len(snapshot.next)==0 :
                break
        
    
        self.root.quit() 
        self.root.destroy() 
        print("Bye!")
        print("All the statistical reports are generated at " + self.conversation_path  + ".")
     



# Define the LLM
llm =  OllamaLLM(model="llama3",temperature=0)

# Thread
thread_p1 = {"configurable": {"thread_id": "1"}}
memory_p1 = MemorySaver()


root = tk.Tk()
root.title("AI Agent for Clinical Research")

# Make the window resizable
root.geometry("1280x960")
root.minsize(300, 200)

# Create an instance of the Agent class

abot = Supervisor(root, llm, memory_p1  )

# Start the keep_asking method in the Tkinter event loop
root.after(1000, abot.run(thread_p1, "1"))

# Start the Tkinter event loop
root.mainloop()




