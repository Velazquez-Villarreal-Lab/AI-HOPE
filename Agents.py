from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd
import pandas.api.types as ptypes
import re
import os
import string
import ast
import Levenshtein


class AgentState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    attributes: list
    messages: Annotated[list, add_messages]

class Supervisor:
    def __init__(self, model, local_memory ):
        print("init")
        graph = StateGraph(AgentState)
        
        graph.add_node("init_Case", self.init_Case_fun)
        graph.add_node("input_data_Case", self.input_data_Case_fun)
        graph.add_node("load_data_Case", self.load_data_Case_fun)
        graph.add_node("init_query_I_Case", self.init_query_I_Case_fun)
        graph.add_node("parse_query_I_Case", self.parse_query_I_Case_fun)
        graph.add_node("init_set_criteria_Case", self.init_set_criteria_Case_fun)
       
        graph.add_node("overview_Case", self.overview_Case_fun)
        graph.add_node("show_attr_values_Case", self.show_attr_values_Case_fun)
        graph.add_node("set_criteria_Case", self.set_criteria_Case_fun)
        graph.add_node("summary_Case", self.summary_Case_fun)

        graph.add_node("init_Ctrl", self.init_Ctrl_fun)
        graph.add_node("input_data_Ctrl", self.input_data_Ctrl_fun)
        graph.add_node("load_data_Ctrl", self.load_data_Ctrl_fun)
        graph.add_node("init_query_I_Ctrl", self.init_query_I_Ctrl_fun)
        graph.add_node("parse_query_I_Ctrl", self.parse_query_I_Ctrl_fun)
        graph.add_node("init_set_criteria_Ctrl", self.init_set_criteria_Ctrl_fun)

        graph.add_node("overview_Ctrl", self.overview_Ctrl_fun)
        graph.add_node("show_attr_values_Ctrl", self.show_attr_values_Ctrl_fun)
        graph.add_node("set_criteria_Ctrl", self.set_criteria_Ctrl_fun)
        graph.add_node("summary_Ctrl", self.summary_Ctrl_fun)

        graph.add_node("exec", self.exec_fun)
        
        graph.add_edge(START, "init_Case")
        graph.add_edge("init_Case", "input_data_Case")
        

        graph.add_conditional_edges(
            "input_data_Case",
            self.make_decision_fun,
            {"1": "load_data_Case","2":"input_data_Case" }
        )

        graph.add_edge("load_data_Case", "init_query_I_Case")
        graph.add_edge("init_query_I_Case", "parse_query_I_Case")
        graph.add_conditional_edges(
            "parse_query_I_Case",
            self.make_decision_fun,
            {"1":"overview_Case", "2": "init_set_criteria_Case","3":"summary_Case" ,"4": "init_query_I_Case"}
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
            {"1": "load_data_Ctrl","2":"input_data_Ctrl" }
        )

        graph.add_edge("load_data_Ctrl", "init_query_I_Ctrl")
        graph.add_edge("init_query_I_Ctrl", "parse_query_I_Ctrl")

        graph.add_conditional_edges(
            "parse_query_I_Ctrl",
            self.make_decision_fun,
            {"1":"overview_Ctrl", "2": "init_set_criteria_Ctrl",  "3":"summary_Ctrl" ,"4": "init_query_I_Ctrl"}
        )
        
        graph.add_edge("init_set_criteria_Ctrl", "set_criteria_Ctrl")
        graph.add_edge("set_criteria_Ctrl", "init_query_I_Ctrl")

        graph.add_edge("overview_Ctrl", "show_attr_values_Ctrl")
        graph.add_edge("show_attr_values_Ctrl", "init_query_I_Ctrl")
                
        graph.add_edge("summary_Ctrl", "exec")

        graph.add_edge("exec", END)
        

        self.graph = graph.compile(
            checkpointer=local_memory,
            interrupt_before=["input_data_Case","input_data_Ctrl"  , "parse_query_I_Case","parse_query_I_Ctrl", "show_attr_values_Case","show_attr_values_Ctrl" , "set_criteria_Case","set_criteria_Ctrl"]
        )
        
        self.model = model
        self.data_repository= []
        self.Case_data_id = ""
        self.Case_criteria_str = []
        self.Case_criteria_logic = []
        self.Ctrl_criteria_str = []
        self.Ctrl_criteria_logic = []

    def find_yes_no_prompt(self, user_input):
        
        str = """
        "Please determine whether the user's input indicates a 'yes' or 'no.' 
        The response should be classified as either 'yes' or 'no' based on the user's input as follows. 
        The output always starts and ends with the square brackets [ and ]. 
        {user_input}
        Example Input:
            Yes, I agree.
        Example Output:
            [yes]
        Example Input:
            No, I don't think so.
        Example Output:
            [no]
        """
        prompt= ChatPromptTemplate.from_template(str)
        chain = prompt | self.model   
        input_dict = {
            "user_input":  user_input
        }
        output = chain.invoke(
            input_dict
        )
        print(output)
        return output

    def find_best_match(self, user_input, word_list):
        
        
        for word in word_list:
            if word.strip() == user_input.strip():
                print("there is a exact match. {}".format(word.strip()))
                return  word.strip() 

        print("there is no exact match. looking for the most similar one.")
        
        
        input_word = user_input.lower()

        # Initialize variables to store the best match and highest similarity score
        best_match = None
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

        - Use 'from' and 'to' to define a range.
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

        - Do not try to correct any type error in the input.
        Example: "Age is 30sss" to (age, 30sss, ==).

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
        print(output)
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
                    print(f"Missing operator between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False  # There is a missing operator
                elif token_types[i] == 'operator':
                    print(f"Missing operand between '{tokens[i -1]}' and '{tokens[i]}'")
                    return False  # There is a missing operand
        # If we reach here, there is no missing operator
        print("No missing operator detected")
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

    def parse_query_prompt(self, messages):
        pt_str ="""
        You are an assistant helping to verify users' intentions.
        You will first classify the input sentences into one of the following 5 categories:

        1. Explore Data - The user is asking to explore the dataset. In this case, the system will provide all available attributes and their values.
        2. Set Up Criteria - The user is defining a criterion to filter or subset the case cohort.
        3. Proceed - The user is asking to move on and go to the next step. All samples in the selected dataset will be included in your case cohort.
        4. The Other - For everything else.
        Output the class of the input sentences at the end of the sentence as your output.
        Your output should always start and end with square brackets [ ].

        Example Input:

        1. Include all samples.
        2. Define Parameters to refine the case cohort.
        3. Let's move on.
        4. Establish Conditions to filter and refine the case cohort.
        5. Show all data attributes.
        6. I am a cat.

        Example Output:

        [3]
        [2]
        [3]
        [2]
        [1]
        [4]
        
        
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
        print(output)
        return output
    
    def return_rownames(self, metadata_df, attr, opr, value_str):
        
        ##step 1 verify attribute 
        sample_list =[]
        attribute_id  = self.find_best_match( attr, metadata_df.columns)
        if attribute_id =="":
            print("[AI] can't find the attribute {} in the metadata.".format(attr))
            return None
        
        print(f"the attribute id is {attribute_id}.")
        #Step 2 verify valid values for non numberic attribute
        value_str = value_str.strip()
        value_list = []
        if value_str.startswith("|") and value_str.endswith("|"):
            print(f"The string '{value_str}' starts and ends with a pipe '|'")
            value_list = [x.strip() for x in value_str.strip('|').split(',')]
        else:
            value_list.append(value_str)
        print(value_list)
        
        for value in value_list:
            if ptypes.is_numeric_dtype(metadata_df[attribute_id]):
                print(f"The column '{attribute_id}' is numeric.")
                max_value = metadata_df[attribute_id].max()
                min_value = metadata_df[attribute_id].min()

                try:
                    value_d = float(value)
                    print(f"The float value is: {value_d}")
                except ValueError:
                    print(f"Error: The string '{value}' cannot be converted to a float.")
                    return None
        
            else:
                print(f"The column '{attribute_id}' is not numeric.")
                unique_values = metadata_df[attribute_id].unique()
                print(unique_values)
                valid_list = []
                for item in unique_values:
                    print(item)
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
                    print(f'{value} is not in [ {", ".join(valid_list)} ] ')
                    return None
        # Step 3 subsetting the metadata df 
        opr = opr.strip() 
        if ptypes.is_numeric_dtype(metadata_df[attribute_id]):
            if opr == ">" :
                sample_list = metadata_df.index[metadata_df[attribute_id] > float(value_list[0])]
            if opr == ">=" :
                sample_list = metadata_df.index[metadata_df[attribute_id] > float(value_list[0])]
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
            if opr == "range":
                value_d_list = [float(x) for x in value_list]
                min_value = min(value_d_list)
                max_value = max(value_d_list)
                print("min_value",min_value)
                print("max_value",max_value)
                matching_row_names =  metadata_df[(metadata_df[attribute_id].astype(float) >= min_value) & (metadata_df[attribute_id].astype(float) <= max_value)].index
                sample_list = matching_row_names.tolist()
        else:
            if opr == ">" or opr == "<" or opr == ">=" or opr == "<=" or opr == "range":
                print(f'the values is not comparable using {opr}')
                return None
            elif opr == "==" or opr == "in":
                sample_list=[]
                for index, item in zip(metadata_df.index, metadata_df[attribute_id]):
                    # print("chk..", index, item)
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    # print("append...", substring.strip(), value_list )
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                # print("append...", substring.strip(), value_list )
                                sample_list.append(index)
                # print(sample_list)
            elif opr == "!=" or opr == "not in":
                sample_list=[]
                for index, item in zip(metadata_df.index, metadata_df[attribute_id]):
                    # print("chk..", index, item)
                    if not isinstance(item, type(pd.NA)):
                        if '|' in item:
                            substrings = item.split('|')
                            for substring in substrings:
                                if substring.strip() in value_list and index not in sample_list :
                                    # print("append...", substring.strip(), value_list )
                                    sample_list.append(index)
                        else:
                            if item.strip() in value_list and index not in sample_list:
                                # print("append...", substring.strip(), value_list )
                                sample_list.append(index)
                sample_set = set(sample_list)
                    # Find the rows (indices) that are in metadata_df.index but not in sample_list
                rows_not_in_sample_list = set(metadata_df.index) - sample_set
                sample_list = list(rows_not_in_sample_list)

        print(f"matched samples {len(sample_list)}")
        return sample_list

    def infix_to_postfix(self, expression):
        output = []
        operator_stack = []
    
        tokens = expression.split()

        for token in tokens:
            print("token : ",token )
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
            print(stack)
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
        
        str = """
        ###################################################################################################
        Step-by-Step Guide to Conduct Case-Control Bioinformatics Data Analysis Using TCGA, GTEx, and CCLE Data
        ###################################################################################################
        Objective: Identify differentially expressed genes and pathways between cancer subtypes and/or normal 
        tissues, and validate these potential therapeutic targets using CCLE (Cancer Cell Line Encyclopedia) 
        data.
        """
        print(str)

        str = """
        Before diving into data analysis, it's crucial to clearly define your case  and control cohorts. 
        This definition is based on clinical attributes and helps ensure that the comparisons you make are 
        meaningful and relevant to your research question.
        
        No need to worry!!! We will guide you through the entire process step by step. You'll have clear 
        instructions and examples along the way to ensure you understand each part of the criteria-setting 
        and analysis. We're here to support you every step of the way.
        """
        print(str)

        str = """
        ###################################################################################################\n
        Let's start by loading the dataset for your case cohort!
        ###################################################################################################\n
        """
        print(str)
        
        str="\n[AI] You have deposite the following data sets in your data folder (data/.)\n"
        print(str)
        directory = "/Users/bearman/Desktop/projects/LLMs/langchain/TCGA_GTEx_CCLE/data/"
        # List all files in the directory
        files = os.listdir(directory)

        # files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        
        for data_id in files: 
            print(data_id.strip())

        self.data_repository = files
      
        pass
        

        
    def load_data_Case_fun(self, state: AgentState):

        metadata_fname = "/Users/bearman/Desktop/projects/LLMs/langchain/TCGA_GTEx_CCLE/data/{}/pt_mut_metadata.tsv".format(self.Case_data_id)
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 )
        # # # Select the first row as keys and the fifth row as values
        # keys = df.iloc[4]
        # values = df.iloc[0]
        # # # Create a dictionary from the selected rows
        # data_dict = dict(zip(keys, values))
    
        # self.Case_metadata_dict = data_dict
        # # # Display the resulting dictionary
        self.Case_metadata_df = df
       
        self.Case_metadata_df = self.Case_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        # print(self.pt_metadata_df.dtypes)
        # column_string = ', '.join(self.pt_metadata_df.columns)
        rows, columns = self.Case_metadata_df.shape
        str = "\n[AI] We identified {} patient samples and {} attributes in the metadata of the {} dataset.\n[AI] We will guide you on how to set criteria based on these attributes to set up your case cohort later.\n".format( rows, columns, self.Case_data_id)
        print(str)

        
        # for key, value in self.Case_metadata_dict.items():
        #     print(f"{key}: {value.replace("#", "")}")
        
        pass
    def input_data_Case_fun(self, state: AgentState):

        messages = state['messages'][-1].content
        
        output = self.find_best_match(messages, self.data_repository  )
        messages = "2"
        print(output)
        if  output==""  :
                print("\n[AI00]***WARNING*** Your input is invalid. Please try again.\n")
        else :
                self.Case_data_id = output
                print("\n[AI] We have detected or inferred that {} is in your input.\n".format(self.Case_data_id))
                
                if self.Case_data_id in self.data_repository :
                    messages = "1"
                else:
                    print("\n[AI] ***WARNING*** Your input is invalid. Please try again.\n")

        return {'messages': [messages]}

        
    def init_query_I_Case_fun(self, state: AgentState):
       
        str = "[AI] You have the following 3 options to proceed from here:\n"
        print(str)
        str ='***Explore Data**:Use this to view all available attributes and their values.\nExamples: "Show data attributes," "Explore the dataset."\n\n***Set Up Criteria***: Define specific criteria to filter the case cohort based on clinical attributes.\nExamples: "Refine the case cohort." "Set up criteria."\n\n***Proceed*** : Indicate you are ready to move on to the next step.\nExamples: "Proceed to the next step," "Move forward with all samples.\n\n"'
        print(str)
        str= '\n\nYou can always type "quit," "exit," or "q" to end the conversation.'
        print(str)
        pass

    def parse_query_I_Case_fun(self, state: AgentState):
        print("parse_query_Case_fun")
        messages = state['messages'][-1].content
        
        output = self.parse_query_prompt(messages)
        
        matches = re.findall(r'\[(.*?)\]', output)
        print(matches)
        data_id_list = []
        output = "4"
        
        for match in matches: 
                data_id_list =  match.split(',')
                if data_id_list != []:
                    for data_id in data_id_list:
                        print(data_id.strip())
                        output = data_id.strip()

        if data_id_list == [] :
            print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

        if data_id_list != []:
            print(len(matches))
            print(len(data_id_list))
    
            if len(matches) > 1 or len(data_id_list)>1 :
                print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
            else :
                if output == "1":
                    print("\n[AI] You want to explore data ...\n" )

                elif output == "2":
                    print("\n[AI] You want to set up criteria to define the case cohort.\n" )

                elif output == "3":
                    print("\n[AI] You want to proceed. All patient samples in the selected dataset will be included in your case cohort.\n" )
                else:
                    print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                # messages = "Yes"

        return {'messages': [output]}
        # return {'messages': [output]}
        
     
    def init_set_criteria_Case_fun(self, state: AgentState):
        os.system('clear')
        str = """
        ###################################################################################################
        Let's define your case cohort!!!
        ###################################################################################################
        """
        print(str)
        str = """
        A valid logic expression is made up of clauses that define relationships between attributes and values 
        using comparison operators. These clauses are connected by "and" or "or" operators, while parentheses 
        help control how the relationships are evaluated. Each clause describes how an attribute relates to a 
        value, which could be a single value, a range, or a set.

        For instance, you can use "is" to express equality (e.g., "Age is 30") or comparison operators like 
        "is greater than" or "is less than" (e.g., "Heart rate is greater than 80"). To define ranges, use 
        "from [Start Value] to [End Value]" (e.g., "Age is from 10 to 20"). To express a set of possible values
        , use "is in" (e.g., "Disease stage is in the set {stage I, stage II, stage III}"). When combining 
        multiple relationships using "and" or "or," parentheses ensure that the intended logical grouping is 
        maintained.

        Without proper use of parentheses, ambiguity can arise. For example, in the expression "Age is greater 
        than 40 or Age is less than 20 and Disease stage is stage II," the intended logic may not be clear. 
        Does the "Disease stage is stage II" apply only to "Age is less than 20," or does it apply to both age 
        conditions? By using parentheses, such as "((Age is greater than 40) or (Age is less than 20)) and 
        (Disease stage is stage II)," the relationships are clearly defined, and the logic is evaluated in 
        the correct order.

        To avoid such confusion, it's best practice to use parentheses around every clause, ensuring clarity 
        and the correct interpretation of the logic. For example: "((Age is greater than 40) or 
        (Age is less than 20)) and (BMI is greater than 25) and (Disease stage is in the set {stage I, stage II, 
        stage III}) and (Gender is male)." This way, the conditions are grouped and interpreted as intended.
        
        """

        print(str)
        
        pass
    def make_decision_fun(self, state: AgentState):
        print("make_decision_Case_fun")
        messages = state['messages'][-1].content
        return messages
        
    
    def show_attr_values_Case_fun(self, state: AgentState):
        print("show_attr_values_Case_fun")
        messages = state['messages'][-1].content
        
        pass
        # return {'messages': [messages]}
    def set_criteria_Case_fun(self, state: AgentState):
        print("set_criteria_Case_fun")
        sample_dict = {}
        output = ""
        messages = state['messages'][-1].content
        messages = messages.replace("{", '|')
        messages = messages.replace("}", '|')
        messages = "("+messages+")"
        if '(' in messages or ')' in messages:
            if not self.check_balanced_parentheses(messages) :
                print("not closed")
                return {'messages': [output]}

        print("here") 
       
        new_expression, condition_map = self.replace_bottom_level_conditions(messages)
        
        print(self.has_valid_operators(new_expression))
        print("Modified Expression:", new_expression)

        cleaned_expression = re.sub(r'[A-Z\s()]+|and|or', '', new_expression)
        if cleaned_expression != '':
            print("not a valid expression?")
            return {'messages': [output]}
            
        if self.check_missing_operator(new_expression) == False:
            print("missing operators")
            return {'messages': [output]}

        if self.has_valid_operators(new_expression) == False:
            print("no_valid_operators")
            return {'messages': [output]}
        
        postorder_list = self.infix_to_postfix(new_expression)
        # postorder_list = [token.lower() if token.lower() in ['and', 'or'] else token for token in postorder_list]

        print(postorder_list)
        print("\nCondition Mapping:")
        for letter, condition in condition_map.items():

            print(f"{letter}: {condition}")
            input_string = self.extract_relationship_prompt(condition)
            # # Regular expression pattern to capture the tuples
            print(input_string)
            # tuple_pattern = r'\"tuples\":\s*\[\s*(\([^]]+?\))\s*\],'
            list_pattern = r'"tuples"\s*:\s*\[\s*([^\]]+)\s*\],'
            # Regular expression pattern to capture the conjunction string
            conjunction_pattern = r'\"conjunction\":\s*\"([^\"]*)\"'

            # Match the pattern against the input string for tuples
            tuple_match = re.search(list_pattern, input_string, re.DOTALL)

            # Match the pattern against the input string for conjunction
            conjunction_match = re.search(conjunction_pattern, input_string)

            print(tuple_match)
            print(conjunction_match)
            if tuple_match is None or conjunction_match is None:
                print("Cannot parse the logic expression!")
                return {'messages': [output]}  
            else:
                 # Extracting the matched tuples string and splitting it by tuple boundaries
               
                tuples_str = tuple_match.group(1).strip()
                print("Extracted tuples:", tuples_str)
                # Regular expression to match tuples across multiple lines
                tuple_pattern = r'\(([^)]+)\)'
                matches = re.findall(tuple_pattern, input_string, re.DOTALL)
                if matches is None:
                    print("Cannot parse the logic expression!")
                    return {'messages': [output]} 
                else:
                    if len(matches) >1:
                        print("more than 1 relationship. use () per relationship.")
                        return {'messages': [output]} 
                    else:
                        for tuple_str in matches:
                            print(tuple_str)
                            token_list = tuple_str.split(",")
                            print("attribute = ", token_list[0])
                            attr = token_list[0]
                            print("opr = ", token_list[-1])
                            opr = token_list[-1]
                            middle_words = ",".join(token_list[1:-1])
                            print("value = ",middle_words)
                            sample_list = self.return_rownames(self.Case_metadata_df, attr,  opr, middle_words)

                            if sample_list is None:
                                print("ㄇㄉㄈㄎ...")
                                return {'messages': [output]} 
                            else:
                                sample_dict[letter] = sample_list
                                


                # tuples_list = ast.literal_eval(tuples_str)
                
    
            # Split the content into individual tuples by finding each occurrence of "(...)"
            #     tuples_list = re.findall(r'\([^)]+\)', tuples_content)
            #     if len(tuples_list) !=1 :
            #         print("not one tuple!")
            #         return {'messages': [output]}  
            # # Loop through each tuple and print it
            #     for tup in tuples_list:
            #         print("Tuple:", tup)


        print(sample_dict)
        sample_list = self.evaluate_postfix(postorder_list, sample_dict )
        print(f"at the end, there are {len(sample_list)} matched samples." )
        print(sample_list)
        return {'messages': [output]}  

    def overview_Case_fun(self, state: AgentState): 
        pass
        
    def summary_Case_fun(self, state: AgentState): 
        pass

    def init_Ctrl_fun(self, state: AgentState):
        print("init_Ctrl_fun")
        messages = state['messages'][-1].content
        print(messages)
        return {'messages': ["init_Ctrl_fun"]}
    
    def input_data_Ctrl_fun(self, state: AgentState):
        pass
    def load_data_Ctrl_fun(self, state: AgentState):
        pass  
    def init_query_I_Ctrl_fun(self, state: AgentState):
        pass
    def parse_query_I_Ctrl_fun(self, state: AgentState): 
        print("parse_query_Ctrl_fun")
        messages = state['messages'][-1].content
        print(messages)
        return {'messages': ["parse_query_Ctrl_fun"]}

    def make_decision_I_Ctrl_fun(self, state: AgentState):
        print("make_decision_Ctrl_fun")
        messages = state['messages'][-1].content
            
        return  "4"
    def init_set_criteria_Ctrl_fun(self, state: AgentState):
        pass

        
    def overview_Ctrl_fun(self, state: AgentState): 
        pass    
    def show_attr_values_Ctrl_fun(self, state: AgentState):
        print("show_attr_values_Ctrl_fun")
        pass

    def set_criteria_Ctrl_fun(self, state: AgentState):
        print("set_criteria_Ctrl_fun")
        pass

    
    def summary_Ctrl_fun(self, state: AgentState): 
        pass
    def exec_fun(self, state: AgentState):   
        print("exec_fun")  
        return {'messages': ["exec_fun"]}

    def run(self,thread):

        
        user_input = ""
           
        for event in self.graph.stream({"messages": ("user", user_input)} ,thread):
            for value in event.values():
                # print(value)
                pass
        snapshot = self.graph.get_state(thread)
        # print(snapshot.values["messages"][-3:])
        # print(snapshot.next[0])
        
        while len(snapshot.next)>0:

            # snapshot = self.graph.get_state(thread)
            # print(snapshot.values["messages"][-3:])
            # print(snapshot.next[0])

            if snapshot.next[0]=="input_data_Case" :
                # print("hi..")
                while True:
                    input_str = input("\n[USER] Please indicate which dataset you would like to include for your case cohort :")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_Case"
                )

            if snapshot.next[0]=="input_data_Ctrl" :
                # print("hi..")
                while True:
                    input_str = input("\n[USER] Please indicate which dataset you would like to include for your control cohort :")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_Case"
                )

            if snapshot.next[0]=="show_attr_values_Case" :
                while True:
                    input_str = input("User Input:")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "overview_Case"
                )
            if snapshot.next[0]=="show_attr_values_Ctrl" :
                while True:
                    input_str = input("User Input:")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "overview_Ctrl"
                )
            
            if snapshot.next[0]=="set_criteria_Case" :
                while True:
                    input_str = input("User Input:")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_set_criteria_Case"
                )

            if snapshot.next[0]=="set_criteria_Ctrl" :
                while True:
                    input_str = input("User Input:")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_set_criteria_Ctrl"
                )

            if snapshot.next[0]=="parse_query_I_Case" :

                while True:
                    input_str = input("\n[USER] What would you like to do next?\n")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_query_I_Case"
                )
            
            if snapshot.next[0]=="parse_query_I_Ctrl" :

                while True:
                    input_str = input("\n[USER] What would you like to do next?\n")
                    if(len(input_str)>0):
                        break
                if input_str.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= "init_query_I_Ctrl"
                )
            
            
            for event in self.graph.stream(None ,thread):
                for value in event.values():
                    pass
            snapshot = self.graph.get_state(thread)
            # print(snapshot.values["messages"][-3:])
            if len(snapshot.next)==0 :
                break
            
                    



# Define the LLM
# llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key,   temperature=0)
llm =  OllamaLLM(model="llama3",temperature=0)

# Thread
thread_p1 = {"configurable": {"thread_id": "1"}}
memory_p1 = MemorySaver()

abot = Supervisor(llm, memory_p1  )

abot.run(thread_p1)


# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
# from PIL import Image
# from io import BytesIO

    
# image_stream = BytesIO(abot.graph.get_graph().draw_mermaid_png())

# # Open the image using PIL
# image = Image.open(image_stream)

# image.save("saved_image.png")
