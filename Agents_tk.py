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
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
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
        
        # Handle window close event (no override needed, allows normal termination)
        self.root.protocol("WM_DELETE_WINDOW",  self.on_close)
        self.conversation_path = ""
        # Create the GUI layout
        self.create_widgets()

    def create_widgets(self):
        # Define a font for the Text and Input widgets
        widget_font = font.Font(family="Helvetica", size=16)
        
        # Define a font for the labels
        label_font = font.Font(family="Helvetica", size=20)

        # Create a label for output with font size 20
        output_label = tk.Label(self.root, text="Conversation:", font=label_font)
        output_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Create a frame to hold the Text widget and the Scrollbar
        self.text_frame = tk.Frame(self.root)
        self.text_frame.grid(row=1, column=0, columnspan=2,padx=5, pady=10, sticky="nsew")

        # Create a Scrollbar
        scrollbar = tk.Scrollbar(self.text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a Text widget with a vertical scrollbar
        self.output_text = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=widget_font)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Attach the Scrollbar to the Text widget
        scrollbar.config(command=self.output_text.yview)

        # Disable the Text widget to make it read-only
        self.output_text.config(state=tk.DISABLED)

        # Configure grid row and column to make the Text widget expand
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create a label for input with font size 20
        input_label = tk.Label(self.root, text='Input: You can type "quit," "exit," or "q" to end the conversation.', font=label_font)
        input_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Input text field (spans column 0)
        self.input_text = tk.Text(self.root, font=widget_font, height=4)
        self.input_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Create a button to submit the input, placed to the right of the input Text widget
        submit_button = tk.Button(self.root, text="Submit", command=self.on_submit)
        submit_button.grid(row=4, column=1, padx=5, pady=5)  # Button next to the Input Text widget

        # Make the input Text widget expand horizontally
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)

        # Create a frame to display the HTML content with HTMLScrolledText
        self.html_frame = tk.Frame(self.root)
        self.html_frame.grid(row=1, column=2, rowspan =3, padx=5, pady=5, sticky="nsew")  # Column 1

        # Create the HTML viewer using HTMLScrolledText
        self.html_viewer = HTMLScrolledText(self.html_frame, width=60, height=30)  # Width and height are set to match the output text
        self.html_viewer.pack(fill=tk.BOTH, expand=True)

        # Ensure the width of output text and HTML viewer are the same
        self.text_frame.config(width=self.html_viewer.winfo_width())

        

    def display_html(self, file_path):
        """
        Load and display HTML content from a file.
        """
        with open(file_path, "r") as file:
            html_content = file.read()

        # Use HTMLScrolledText to display HTML content
        self.html_viewer.set_html(html_content)

    def on_submit(self):
        
        input_str = self.input_text.get("1.0", tk.END).strip()  # Get text from line 1, char 0 to the end

        self.output_text.config(state=tk.NORMAL)
            # Append user input to the output_text widget
        self.output_text.insert(tk.END, f"[AI] Your input is : {input_str}\n")
            
        self.output_text.insert(tk.END, "[AI] Processing your input ...\n")
            # Disable the Text widget again to make it read-only
        self.output_text.config(state=tk.DISABLED)
            # Auto-scroll to the latest line
        self.output_text.see(tk.END)
      
        # Check if the input string is empty
        if not input_str:
            # self.tk_print("***WARNING!!!***\nYour input is empty! Please provide your commands in the text box below!!\n")
            return  # Exit the function if input is empty

        
        # Replace all newline characters with spaces
        input_str = input_str.replace("\n", " ")
        # Set the modified input string
        self.user_input.set(input_str)
        # Clear the input text area after submission
        self.input_text.delete("1.0", tk.END)  # Clear the input text area after submission


    def on_close(self):
        # This method is called when the user clicks the close button
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # self.root.quit()  # Stop the Tkinter main loop
            # self.root.destroy()  # Destroy the root window, which terminates the application
            self.user_input.set('q')

    def tk_print(self, input_str):
        # Test whether the input can be converted to a string
        try:
        # Attempt to convert the input to a string
            input_str = str(input_str)
        except (ValueError, TypeError):
        # If conversion fails, do nothing and return
            return

        # Enable the Text widget to insert new content
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

        # Initialize variables to store the best match and highest similarity score
        best_match = ""
        highest_similarity = -1

        # Iterate over the word list
        for word in word_list:
            # Convert each word in the list to lowercase
            word = word.strip()
            tmp_word = word.lower()
            tmp_word = tmp_word.strip()

            # Calculate the Levenshtein distance
            distance = Levenshtein.distance(input_word, tmp_word)

            # Calculate similarity as 1 - (distance / max length of the two words)
            max_len = max(len(input_word), len(word))
            similarity = 1 - (distance / max_len)

            # Update best match if this word has a higher similarity
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
        "=="   (Equal to)
        "!="  (Not equal to)
        ">"    (Greater than)
        "<"    (Less than)
        ">="  (Greater than or equal to)
        "<="   (Less than or equal to)
        "range" (Define a range)
        "in"  (Define a set of values)
        "not in"  (Exclude a set of values)

        - For "less than or equal to," the operator is <= and the value must be a number.
        Example: "BMI is less than or equal to 20" to (BMI, 20, <=).
        
        - For "greater than or equal to," the operator is >= and the value must be a number.
        Example: "BMI is greater than or equal to 20" to (BMI, 20, >=).

        - Use only 'from' and 'to' to define a range.
        Example: "Age is from 10 to 20" to (age, |10, 20|, range).

        - A range always starts and ends with a pipe '|'  

        - Use 'in' or 'not in' to define a set of values. Keep the values as the same as they appear in the input
        Example: "The humidity is in the set (50, 60, 70)" to (humidity, |50, 60, 70|, in).
        Example: "Stage is not in the set (50, 60, 70)" to (humidity, |50, 60, 70|, not in).
        Example: "Tumor stage is in the set (stage I, stage II and IV)." to (age, |stage I, stage II, IV|, in)

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
        
        - If "is not" or "is no" appears in the input, interpret the value as the word after "not" or "no" as the value for the relationship. "not" and "no" is not included in the value.
        Example: "APC_mutation_effect is not not_identifiable" should have a value of "not identifiable."

        - Do not change the value in the input.
        Example: "Age is 30sss" to (age, 30sss, ==).
        Example: "Age is not not_identifiable" to (age, not_identifiable, !=).

        - Each relationship should be represented as a tuple in the format (variable, value, comparison operator). 
        - Tuples should always start and end with parentheses ( ).

        ### Step 2: Determine the Conjunction
        If multiple relationships (tuples) are connected by "and" or "or," specify the conjunction in the output.  
        If the sentence contains only one tuple, leave the conjunction as an empty string ("").  
        If no conjunction is explicitly mentioned, assume "and" by default.

        ### Step 3: Output as JSON Object
        The output should be a JSON object only with two fields:
        1. "tuples": A list containing all the extracted tuples, where each tuple starts and ends with parentheses ( ), and the values are formatted according to the rules above.
        2. "conjunction": A string that indicates whether the relationships are connected by "and," "or," or an empty string for single tuples.

        Input:
        {user_input}

        Output the JSON only. No explanation or other text.
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
        # self.tk_print(output)
        return output
    
    def replace_bottom_level_conditions(self,expression):
    
        letters =  iter(
            list(string.ascii_uppercase) + 
            list(string.ascii_lowercase) + 
            [u + u for u in string.ascii_uppercase] + 
            [l + l for l in string.ascii_lowercase]
        )   
        condition_dict = {}

        # Regex pattern to find the innermost clauses (i.e., no nested parentheses inside)
        pattern = r'\([^()]+\)'

        # Replace the innermost conditions with letters
    
        # expression = replace_innermost(expression, condition_dict, letters)
    
        matches = re.findall(pattern, expression)
    
        for match in matches:
        # Strip leading/trailing whitespace
            condition = match.strip()
        
        # Assign a letter to the condition
            letter = next(letters)
        
        # Store the condition in the dictionary
            condition_dict[letter] = condition
        
        # Replace the condition in the expression with the letter
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
        # Now check for consecutive operands or operators
        for i in range(1, len(token_types)):
            if token_types[i] == token_types[i - 1]:
                if token_types[i] == 'operand':
                    self.tk_print(f"Missing operator between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False  # There is a missing operator
                elif token_types[i] == 'operator':
                    self.tk_print(f"Missing operand between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False  # There is a missing operand
        # If we reach here, there is no missing operator
        self.tk_print("No missing operator detected")
        return True

    def has_valid_operators(self, expression):
        # Remove spaces for easier parsing
        expression = expression.replace(" ", "")
    
        # Use a regular expression to split the expression into tokens (letters, parentheses, operators)
        tokens = re.findall(r'[A-Z]|\(|\)|and|or', expression)
    
        # Operators that we are checking
        operators = {"and", "or"}
    
        prev_token = None
    
        for i, token in enumerate(tokens):
            if token in operators:
                # Check if the previous token is a valid operand (letter or closing parenthesis)
                if prev_token is None or prev_token in operators or prev_token == '(':
                    return False  # Invalid operator placement (no valid operand before the operator)
            
                # After the operator, the next token must also be a valid operand (letter or opening parenthesis)
                if i == len(tokens) - 1:
                    return False  # If operator is at the end of the expression, it's invalid
                next_token = tokens[i + 1]
                if next_token in operators or next_token == ')':
                    return False  # Invalid operator placement (no valid operand after the operator)
        
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
        # self.tk_print(output)
        return output
    
    def run_script(self, script_fname, arg_fname ):
        command = ['python3', script_fname, arg_fname]
    
        # Run the subprocess
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Script executed successfully.")
            print(result.stdout)  # Print output from the script
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
        self.tk_print(output)
        return output
    


    def return_rownames(self, metadata_df, attr, opr, value_str):
        attr = attr.replace('"', "")
        opr = opr.replace('"', "")
        value_str = value_str.replace('"', "")
        ##step 1 verify attribute 
        sample_list =[]
        attribute_id  = self.find_best_match( attr, metadata_df.columns)
        if attribute_id =="":
            self.tk_print("[AI] I can't find the attribute {} in the metadata.".format(attr))
            return None
        
        self.tk_print(f"[AI] {attribute_id} is used here.")
        #Step 2 verify valid values for non numberic attribute
        value_str = value_str.strip()
        value_list = []
        if value_str.startswith("|") and value_str.endswith("|"):
            # self.tk_print(f"The string '{value_str}' starts and ends with a pipe '|'")
            value_list = [x.strip() for x in value_str.strip('|').split(',')]
        else:
            value_list.append(value_str)
        # self.tk_print(value_list)
        
        for value in value_list:
            if ptypes.is_numeric_dtype(metadata_df[attribute_id]):
                # self.tk_print(f"The column '{attribute_id}' is numeric.")
                max_value = metadata_df[attribute_id].max()
                min_value = metadata_df[attribute_id].min()

                try:
                    value_d = float(value)
                    # self.tk_print(f"The float value is: {value_d}")
                except ValueError:
                    self.tk_print(f"[AI] Error: The string '{value}' cannot be converted to a float.")
                    return None
        
            else:
                # self.tk_print(f"The column '{attribute_id}' is not numeric.")
                unique_values = metadata_df[attribute_id].unique()
                # self.tk_print(unique_values)
                valid_list = []
                for item in unique_values:
                    # self.tk_print(item)
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() not in valid_list:
                                    valid_list.append(substring.strip())
                        else:
                            if item not in valid_list:
                                valid_list.append(item)

                if not (value in valid_list):
                    self.tk_print(f'[AI] {value} is not in [ {", ".join(valid_list)} ]. Use "explore data" to check valid values for the data attributes.')
                    return None
        # Step 3 subsetting the metadata df 
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
                    # Find the rows (indices) that are in metadata_df.index but not in sample_list
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
                    # self.tk_print("chk..", index, item)
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    # self.tk_print("append...", substring.strip(), value_list )
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                # self.tk_print("append...", substring.strip(), value_list )
                                sample_list.append(index)
                # self.tk_print(sample_list)
            elif opr == "!=" or opr == "not in":
                sample_list=[]
                for index, item in zip(metadata_df.index, metadata_df[attribute_id]):
                    # self.tk_print("chk..", index, item)
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    # self.tk_print("append...", substring.strip(), value_list )
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                # self.tk_print("append...", substring.strip(), value_list )
                                sample_list.append(index)
                sample_set = set(sample_list)
                    # Find the rows (indices) that are in metadata_df.index but not in sample_list
                rows_not_in_sample_list = set(metadata_df.index) - sample_set
                sample_list = list(rows_not_in_sample_list)

        # self.tk_print(f"matched samples {len(sample_list)}")
        return sample_list

    def infix_to_postfix(self, expression):
        output = []
        operator_stack = []
    
        tokens = expression.split()

        for token in tokens:
            # self.tk_print(f"token : {token}" )
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()  # Pop '('
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
        # Initialize an empty stack
        stack = []
        i=0
        for token in expression:
            # self.tk_print(stack)
            token= token.strip()

            if token in sample_dict.keys():  # If the token is a letter, get its value from the dictionary
                stack.append(token)
            elif token == 'or':  # Pop two operands, evaluate addition, push result
                operand2 = stack.pop()
                operand1 = stack.pop()
                list1 = sample_dict[operand1]
                list2 = sample_dict[operand2]
                union_list = list(set(list1) | set(list2))
                sample_dict["$"+str(i)] = union_list
                stack.append("$"+str(i))
                i=i+1
            elif token == 'and':  # Pop two operands, evaluate multiplication, push result
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
        # Ensure all strings are within the specified character limits
        df['Name'] = df['Name'].str.slice(0, first_col_width)
        df['Description'] = df['Description'].str.slice(0, second_col_width)


        # Print the column headers with specified widths
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
        
        # Split the file content into lines and process each line
        for key in df.index:
            # key, value = line.split("=", 1)  # Split each line into key and value at "="
            config_dict[key] = df.loc[ key, "value" ]  # Assign key and value to the dictionary
        # print(config_dict)
        
        metadata_fname = "data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] )
        self.Case_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
        self.Case_config_dict = config_dict
        self.Case_metadata_df = df
        print(self.Case_metadata_df.dtypes)
        self.Case_metadata_df = self.Case_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        print(self.Case_metadata_df.dtypes)
        # column_string = ', '.join(self.pt_metadata_df.columns)
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
        # print("match:" + data_attr)
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
        # self.tk_print("set_criteria_Case_fun")
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
        
        # self.tk_print(self.has_valid_operators(new_expression))
        # self.tk_print(new_expression)

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
        # postorder_list = [token.lower() if token.lower() in ['and', 'or'] else token for token in postorder_list]

        # self.tk_print(postorder_list)
        # self.tk_print("\nCondition Mapping:")
        for letter, condition in condition_map.items():

            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)
            # # Regular expression pattern to capture the tuples
            # self.tk_print(f'[AI] I think it means {input_string}')
            # tuple_pattern = r'\"tuples\":\s*\[\s*(\([^]]+?\))\s*\],'
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            # Regular expression pattern to capture the conjunction string
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            # Match the pattern against the input string for tuples
            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            # Match the pattern against the input string for conjunction
            conjunction_match = re.search(conjunction_pattern, input_string)

            # self.tk_print(tuple_match)
            # self.tk_print(conjunction_match)
            if tuple_match is None or conjunction_match is None:
                self.tk_print("[AI] Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
                 # Extracting the matched tuples string and splitting it by tuple boundaries
               
                tuples_str = tuple_match.group(1).strip()
                # self.tk_print(  tuples_str)
                # Regular expression to match tuples across multiple lines
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
                            # self.tk_print(tuple_str)
                            token_list = tuple_str.split(",")
                            # self.tk_print("attribute = ", token_list[0])
                            attr = token_list[0]
                            # self.tk_print("opr = ", token_list[-1])
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            # self.tk_print("value = ",middle_words)
                            self.tk_print(f'[AI] I think it means "{attr} {opr} {middle_words}".' )
                            sample_list = self.return_rownames(self.Case_metadata_df, attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No sample matches this criteria. "+tuple_str)

                                return {'messages': [output]} 
                            else:
                                sample_dict[letter] = sample_list
                                

        # self.tk_print(sample_dict)
        sample_list = self.evaluate_postfix(postorder_list, sample_dict )
        
        # self.tk_print(sample_list)
        self.Case_sample_ids = sample_list
         ### summerize data selection here
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
        # Ensure all strings are within the specified character limits
        df['Name'] = df['Name'].str.slice(0, first_col_width)
        df['Description'] = df['Description'].str.slice(0, second_col_width)


        # Print the column headers with specified widths
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
        
        # Split the file content into lines and process each line
        for key in df.index:
            # key, value = line.split("=", 1)  # Split each line into key and value at "="
            config_dict[key] = df.loc[ key,  "value"  ]  # Assign key and value to the dictionary
        # print(config_dict)
        metadata_fname = "data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] )
        self.Ctrl_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
    
        self.Ctrl_metadata_df = df
        self.Ctrl_config_dict = config_dict
        self.Ctrl_metadata_df = self.Ctrl_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        # self.tk_print(self.pt_metadata_df.dtypes)
        # column_string = ', '.join(self.pt_metadata_df.columns)
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
        # print("match:" + data_attr)
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
        
        # self.tk_print(self.has_valid_operators(new_expression))
        # self.tk_print(new_expression)

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
        # postorder_list = [token.lower() if token.lower() in ['and', 'or'] else token for token in postorder_list]

        # self.tk_print(postorder_list)
        # self.tk_print("\nCondition Mapping:")
        for letter, condition in condition_map.items():

            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)
            # # Regular expression pattern to capture the tuples
            # self.tk_print(f'[AI] I think it means {input_string}')
            # tuple_pattern = r'\"tuples\":\s*\[\s*(\([^]]+?\))\s*\],'
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            # Regular expression pattern to capture the conjunction string
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            # Match the pattern against the input string for tuples
            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            # Match the pattern against the input string for conjunction
            conjunction_match = re.search(conjunction_pattern, input_string)

            # self.tk_print(tuple_match)
            # self.tk_print(conjunction_match)
            if tuple_match is None or conjunction_match is None:
                self.tk_print("[AI] Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
                 # Extracting the matched tuples string and splitting it by tuple boundaries
               
                tuples_str = tuple_match.group(1).strip()
                # self.tk_print(  tuples_str)
                # Regular expression to match tuples across multiple lines
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
                            # self.tk_print(tuple_str)
                            token_list = tuple_str.split(",")
                            # self.tk_print("attribute = ", token_list[0])
                            attr = token_list[0]
                            # self.tk_print("opr = ", token_list[-1])
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            # self.tk_print("value = ",middle_words)
                            self.tk_print(f'[AI] I think it means "{attr} {opr} {middle_words}".' )
                            sample_list = self.return_rownames(self.Ctrl_metadata_df, attr,  opr, middle_words)

                            if sample_list is None:
                                self.tk_print("[AI] No sample matches this criteria. "+tuple_str)

                                return {'messages': [output]} 
                            else:
                                sample_dict[letter] = sample_list
                                

        # self.tk_print(sample_dict)
        sample_list = self.evaluate_postfix(postorder_list, sample_dict )
        
        # self.tk_print(sample_list)
        self.Ctrl_sample_ids = sample_list
         ### summerize data selection here
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
        # messages = state['messages'][-1].content
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
        # postorder_list = [token.lower() if token.lower() in ['and', 'or'] else token for token in postorder_list]

        # self.tk_print(postorder_list)
        # self.tk_print("\nCondition Mapping:")
        for letter, condition in condition_map.items():

            # self.tk_print(f"{letter}: {condition}")
            self.tk_print(f'[AI] I am reasoning what {condition} means')
            input_string = self.extract_relationship_prompt(condition)
            # # Regular expression pattern to capture the tuples
            # self.tk_print(input_string)
            # tuple_pattern = r'\"tuples\":\s*\[\s*(\([^]]+?\))\s*\],'
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            # Regular expression pattern to capture the conjunction string
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            # Match the pattern against the input string for tuples
            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            # Match the pattern against the input string for conjunction
            conjunction_match = re.search(conjunction_pattern, input_string)

            self.tk_print(tuple_match)
            self.tk_print(conjunction_match)
            if tuple_match is None or conjunction_match is None:
                self.tk_print("Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
                 # Extracting the matched tuples string and splitting it by tuple boundaries
               
                tuples_str = tuple_match.group(1).strip()
                self.tk_print(  tuples_str)
                # Regular expression to match tuples across multiple lines
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
                            # self.tk_print("attribute = ", token_list[0])
                            attr = token_list[0]
                            # self.tk_print("opr = ", token_list[-1])
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            # self.tk_print("value = ",middle_words)
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


        # Define the 2x2 table
        #      |in context | Out of context |
        # Case |     a     |     b          |
        # Ctrl |     c     |     d          |

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
                    self.tk_print("\n[AI] You want to test multi\n" )
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
        print("multiple_Survival")
        messages = state['messages'][-1].content
        print(messages)
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
        
        # output = ",".join(self.surv_extra)
        # return {'messages': [output]}  

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
        
        ## check whether if OS is defined 
        OS_flag =1 
        if "OS_TIME" in self.Case_config_dict and self.Case_config_dict["OS_TIME"].strip() !="":
            print("OS_TIME exists and has a non-empty value:", self.Case_config_dict["OS_TIME"])
        else:
            OS_flag=0
            print("OS_TIME is either not present or has an empty value.")

        if "OS_TIME" in self.Ctrl_config_dict and self.Ctrl_config_dict["OS_TIME"].strip() !="":
            print("OS_TIME exists and has a non-empty value:", self.Ctrl_config_dict["OS_TIME"])
        else:
            OS_flag=0
            print("OS_TIME is either not present or has an empty value.")
        
        if "OS_STATUS" in self.Case_config_dict and self.Case_config_dict["OS_STATUS"].strip() !="":
            print("OS_STATUS exists and has a non-empty value:", self.Case_config_dict["OS_STATUS"])
        else:
            OS_flag=0
            print("OS_STATUS is either not present or has an empty value.")

        if "OS_STATUS" in self.Ctrl_config_dict and self.Ctrl_config_dict["OS_STATUS"].strip() !="":
            print("OS_STATUS exists and has a non-empty value:", self.Ctrl_config_dict["OS_STATUS"])
        else:
            OS_flag=0
            print("OS_STATUS is either not present or has an empty value.")
        
        ## if OS data exist
        if OS_flag ==1:
            msg_dict["output_OS_png"]=self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png"
            msg_dict["Case_OS_TIME"] = self.Case_config_dict["OS_TIME"] 
            msg_dict["Case_OS_STATUS"] = self.Case_config_dict["OS_STATUS"] 
            msg_dict["Ctrl_OS_TIME"] = self.Ctrl_config_dict["OS_TIME"] 
            msg_dict["Ctrl_OS_STATUS"] = self.Ctrl_config_dict["OS_STATUS"] 

        ## check whether if PFS
        
        PFS_flag =1 
        if "PFS_TIME" in self.Case_config_dict and self.Case_config_dict["PFS_TIME"].strip() !="":
            print("PFS_TIME exists and has a non-empty value:", self.Case_config_dict["PFS_TIME"])
        else:
            PFS_flag=0
            print("PFS_TIME is either not present or has an empty value.")

        if "PFS_TIME" in self.Ctrl_config_dict and self.Ctrl_config_dict["PFS_TIME"].strip() !="":
            print("PFS_TIME exists and has a non-empty value:", self.Ctrl_config_dict["PFS_TIME"])
        else:
            PFS_flag=0
            print("PFS_TIME is either not present or has an empty value.")
        
        if "PFS_STATUS" in self.Case_config_dict and self.Case_config_dict["PFS_STATUS"].strip() !="":
            print("PFS_STATUS exists and has a non-empty value:", self.Case_config_dict["PFS_STATUS"])
        else:
            PFS_flag=0
            print("PFS_STATUS is either not present or has an empty value.")

        if "PFS_STATUS" in self.Ctrl_config_dict and self.Ctrl_config_dict["PFS_STATUS"].strip() !="":
            print("PFS_STATUS exists and has a non-empty value:", self.Ctrl_config_dict["PFS_STATUS"])
        else:
            PFS_flag=0
            print("PFS_STATUS is either not present or has an empty value.")
        
        ## if OS data exist
        if PFS_flag ==1:
            msg_dict["output_PFS_png"]=self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png"
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 
            msg_dict["Ctrl_PFS_TIME"] = self.Ctrl_config_dict["PFS_TIME"] 
            msg_dict["Ctrl_PFS_STATUS"] = self.Ctrl_config_dict["PFS_STATUS"] 
        

        ## check additional attributes are valid
        
        msg_dict["EXTRA_ATTR"] = self.surv_extra
        

        with open( self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl", 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        self.run_script( "SURV_Agent.py",self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" )
        self.html_fname = self.conversation_path+"/Surv_"+str(self.surv_num) +".html"

        self.surv_num = self.surv_num+1
        # return {'messages': [output]}  
        
    def run(self,thread, thread_id):

        self.thread_id = thread_id
        user_input = ""
        current_directory = os.getcwd()
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(current_directory+"/conversations/"+current_time)
     
        self.conversation_path = current_directory+"/conversations/"+current_time
        for event in self.graph.stream({"messages": ("user", user_input)} ,thread):
            for value in event.values():
                # self.tk_print(value)
                pass
        snapshot = self.graph.get_state(thread)
        
        while len(snapshot.next)>0:

            # Read the entire file into a string
            
            conversation_content = "\n".join(self.conversation_buffer)
            self.output_text.config(state=tk.NORMAL)
            # Append user input to the output_text widget
            self.output_text.insert(tk.END, conversation_content+"\n")
            # Disable the Text widget again to make it read-only
            self.output_text.config(state=tk.DISABLED)
            # Auto-scroll to the latest line
            self.output_text.see(tk.END)
            
            self.conversation_buffer=[]
            
            self.display_html(self.html_fname)

            self.user_input.set('')  # Clear the user_input variable
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
                # while True:
                #     input_str = input("User Input:")
                #     if(len(input_str)>0):
                #         break
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
            # self.tk_print(snapshot.values["messages"][-3:])
            if len(snapshot.next)==0 :
                break

        self.root.quit() 
        self.root.destroy() 
     



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

print("Bye!")


from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image
from io import BytesIO

    
image_stream = BytesIO(abot.graph.get_graph().draw_mermaid_png())

# Open the image using PIL
image = Image.open(image_stream)

image.save("saved_image.png")
