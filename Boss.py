from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from typing import Annotated
from typing_extensions import TypedDict

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

class Boss:
    def __init__(self,  model, local_memory ):
        graph = StateGraph(AgentState)

        graph.add_node("initQ", self.initQ_fun)
        graph.add_node("init_Qtypechk_1", self.init_Qtypechk_1_fun)
        graph.add_node("init_Qtypechk_2", self.init_Qtypechk_2_fun)

        graph.add_node("init_pred_twoGroups", self.init_pred_twoGroups_fun)
        graph.add_node("pred_twoGroups", self.__chk_yn_fun)
        
        graph.add_node("init_switch_twoGroups", self.init_switch_twoGroups_fun)
        graph.add_node("switch_twoGroups", self.__chk_yn_fun)
        
        graph.add_node("init_pred_lookup", self.init_pred_lookup_fun) 
        graph.add_node("pred_lookup", self.__chk_yn_fun)
        
        graph.add_node("init_switch_lookup", self.init_switch_lookup_fun)
        graph.add_node("switch_lookup", self.__chk_yn_fun)
         
        graph.add_node("init_lookup", self.init_lookup_fun)

        graph.add_node("input_data_lookup", self.input_data_Case_fun)
        graph.add_node("load_data_lookup", self.load_data_Case_fun)
        graph.add_node("summary_data_lookup",  self.__chk_yn_fun)

        graph.add_node("init_decide_subset_lookup",  self.init_decide_subset_lookup_fun)
        graph.add_node("decide_subset_lookup",  self.__chk_yn_fun)

        graph.add_node("init_query_I_lookup", self.init_query_I_Case_fun)
        graph.add_node("parse_query_I_lookup", self.parse_query_I_fun)
        
        graph.add_node("init_set_criteria_lookup", self.init_set_criteria_fun)
        graph.add_node("set_criteria_lookup", self.set_criteria_Case_fun)
        
        graph.add_node("overview_lookup", self.overview_Case_fun)
        graph.add_node("show_attr_values_lookup", self.show_attr_values_Case_fun)
        
        graph.add_node("summary_lookup",  self.summary_Case_fun)

        graph.add_node("init_exec_lookup", self.init_exec_fun)
        graph.add_node("pred_surv_lookup", self.pred_surv_fun)
        graph.add_node("pred_data_lookup", self.pred_data_fun)

        graph.add_node("chk_surv_lookup", self.__chk_yn_fun)
        graph.add_node("chk_data_lookup", self.__chk_yn_fun)
        
        graph.add_node("init_VOI_lookup", self.init_VOI_lookup_fun)
        graph.add_node("parse_VOI_lookup", self.parse_VOI_lookup_fun)

        graph.add_node("init_RL_lookup", self.init_RL_fun)

        graph.add_node("init_wRL_lookup", self.init_wRL_fun)
        graph.add_node("init_woRL_lookup", self.init_woRL_fun)

        graph.add_node("parse_wRL_lookup", self.__chk_yn_fun)
        graph.add_node("parse_woRL_lookup", self.__chk_yn_fun)

        graph.add_node("init_CL_lookup", self.init_CL_fun)
        graph.add_node("parse_CL_lookup", self.parse_CL_fun)

        graph.add_node("all_CL_lookup", self.all_CL_lookup_fun)
        graph.add_node("one_CL_lookup", self.one_CL_lookup_fun)
        graph.add_node("simple_CL_lookup", self.simple_CL_lookup_fun)

        graph.add_node("init_surv_RL_lookup", self.init_RL_fun)

        graph.add_node("init_surv_wRL_lookup", self.init_wRL_fun)
        graph.add_node("init_surv_woRL_lookup", self.init_woRL_fun)

        graph.add_node("parse_surv_wRL_lookup", self.__chk_yn_fun)
        graph.add_node("parse_surv_woRL_lookup", self.__chk_yn_fun)


        graph.add_node("init_surv_CL_lookup", self.init_CL_fun)
        graph.add_node("parse_surv_CL_lookup", self.parse_CL_fun)

        graph.add_node("all_surv_lookup", self.all_surv_lookup_fun)
        graph.add_node("one_surv_lookup", self.one_surv_lookup_fun)
        graph.add_node("simple_surv_lookup", self.simple_surv_lookup_fun)
        ####


        graph.add_node("init_Case", self.init_Case_fun)
        graph.add_node("input_data_Case", self.input_data_Case_fun)
        graph.add_node("load_data_Case", self.load_data_Case_fun)
        graph.add_node("summary_data_Case",  self.__chk_yn_fun)

        graph.add_node("init_query_I_Case", self.init_query_I_Case_fun)
        graph.add_node("parse_query_I_Case", self.parse_query_I_fun)
        
        graph.add_node("init_set_criteria_Case", self.init_set_criteria_fun)
        graph.add_node("set_criteria_Case", self.set_criteria_Case_fun)
        
        graph.add_node("overview_Case", self.overview_Case_fun)
        graph.add_node("show_attr_values_Case", self.show_attr_values_Case_fun)
        
        graph.add_node("summary_Case",  self.summary_Case_fun)
        

        graph.add_node("init_Ctrl", self.init_Ctrl_fun)
        graph.add_node("input_data_Ctrl", self.input_data_Ctrl_fun)
        graph.add_node("load_data_Ctrl", self.load_data_Ctrl_fun)
        graph.add_node("summary_data_Ctrl",  self.__chk_yn_fun)

        graph.add_node("init_query_I_Ctrl", self.init_query_I_Ctrl_fun)
        graph.add_node("parse_query_I_Ctrl", self.parse_query_I_fun)
        
        graph.add_node("init_set_criteria_Ctrl", self.init_set_criteria_fun)
        graph.add_node("set_criteria_Ctrl", self.set_criteria_Ctrl_fun)
        
        graph.add_node("overview_Ctrl", self.overview_Ctrl_fun)
        graph.add_node("show_attr_values_Ctrl", self.show_attr_values_Ctrl_fun)
        graph.add_node("summary_Ctrl", self.summary_Ctrl_fun)

        graph.add_node("finalize_data", self.__chk_yn_fun)

        graph.add_node("init_exec", self.init_exec_fun)
        graph.add_node("pred_surv", self.pred_surv_fun)
        graph.add_node("pred_data", self.pred_data_fun)

        graph.add_node("chk_surv", self.__chk_yn_fun)
        graph.add_node("chk_data", self.__chk_yn_fun)
        
        graph.add_node("init_compare", self.init_compare_fun)
        graph.add_node("chk_compare", self.__chk_yn_fun)

        graph.add_node("init_CL", self.init_CL_fun)
        graph.add_node("parse_CL", self.parse_CL_fun)


        graph.add_node("chk_CL", self.chk_CL_fun)

        graph.add_node("all_CL", self.all_CL_fun)
        # graph.add_node("init_one_CL", self.init_one_CL_fun)
        graph.add_node("one_CL", self.one_CL_fun)


        graph.add_node("init_OR", self.init_OR_fun)
        graph.add_node("parse_OR", self.parse_OR_fun)

        graph.add_node("init_Survival", self.init_Survival_fun)
        graph.add_node("parse_Survival", self.__chk_yn_fun)
        graph.add_node("init_multiple_Survival", self.init_multiple_Survival_fun)
        graph.add_node("multiple_Survival", self.multiple_Survival_fun)
        graph.add_node("run_Survival", self.run_Survival_fun)

        graph.add_node("init_end", self.init_end_fun)

        graph.add_node("parse_end", self.__chk_yn_fun)

        graph.add_edge(START, "initQ")
        graph.add_edge("initQ", "init_Qtypechk_1")
        graph.add_conditional_edges(
            "init_Qtypechk_1",
            self.make_decision_fun,
            {1: "init_Qtypechk_2", 2:"initQ" }
        )

        graph.add_conditional_edges(
            "init_Qtypechk_2",
            self.make_decision_fun,
            {1: "init_pred_twoGroups", 2:"init_pred_lookup" }
        )
        graph.add_edge("init_pred_twoGroups", "pred_twoGroups") 
        graph.add_edge("init_pred_lookup", "pred_lookup") 

        graph.add_conditional_edges(
            "pred_twoGroups",
            self.make_decision_fun,
            {1: "init_Case" , 2:"init_switch_twoGroups",3:"pred_twoGroups"}
        )

        graph.add_conditional_edges(
            "pred_lookup",
            self.make_decision_fun,
            {1: "init_lookup" , 2:"init_switch_lookup" ,3:"pred_lookup"}
        )

        graph.add_edge("init_switch_twoGroups", "switch_twoGroups") 
        graph.add_edge("init_switch_lookup", "switch_lookup") 

        graph.add_conditional_edges(
            "switch_twoGroups",
            self.make_decision_fun,
            {1: "init_lookup" , 2:"initQ" ,3:"switch_twoGroups"}
        )

        graph.add_conditional_edges(
            "switch_lookup",
            self.make_decision_fun,
            {1: "init_Case" , 2:"initQ" ,3:"switch_lookup"}
        )
        

        ###
        graph.add_edge("init_lookup", "input_data_lookup")

        graph.add_conditional_edges(
            "input_data_lookup",
            self.make_decision_fun,
            {1: "load_data_lookup", 2:"input_data_lookup" }
        )
        
    

        graph.add_edge("load_data_lookup", "summary_data_lookup")

        graph.add_conditional_edges(
            "summary_data_lookup",
            self.make_decision_fun,
            {1: "init_decide_subset_lookup" , 2:"init_lookup" ,3:"summary_data_lookup"}
        )

        graph.add_edge("init_decide_subset_lookup", "decide_subset_lookup")
        graph.add_conditional_edges(
            "decide_subset_lookup",
            self.make_decision_fun,
            {1: "init_exec_lookup" , 2:"init_query_I_lookup" ,3:"decide_subset_lookup"}
        )

        graph.add_edge("init_query_I_lookup", "parse_query_I_lookup")

        graph.add_conditional_edges(
            "parse_query_I_lookup",
            self.make_decision_fun,
            {1:"overview_lookup", 2: "init_set_criteria_lookup",3:"summary_lookup" ,4: "init_query_I_lookup"}
        )

        graph.add_edge("init_set_criteria_lookup", "set_criteria_lookup")
        graph.add_edge("set_criteria_lookup", "init_query_I_lookup")
        graph.add_edge("overview_lookup", "show_attr_values_lookup")
        graph.add_edge("show_attr_values_lookup", "init_query_I_lookup")

        graph.add_edge("summary_lookup", "init_exec_lookup")
        graph.add_conditional_edges(
            "init_exec_lookup",
            self.make_decision_fun,
            {1:"pred_surv_lookup", 2:"pred_data_lookup"}
        )
        graph.add_edge("pred_surv_lookup", "chk_surv_lookup")
        graph.add_edge("pred_data_lookup", "chk_data_lookup")
        graph.add_conditional_edges(
            "chk_surv_lookup",
            self.make_decision_fun,
            {1: "init_surv_RL_lookup", 2:"init_VOI_lookup",3:"chk_surv_lookup" }
        )
        
        graph.add_conditional_edges(
            "init_surv_RL_lookup",
            self.make_decision_fun,
            {1: "init_surv_wRL_lookup", 2:"init_surv_woRL_lookup" }
        )

        graph.add_edge("init_surv_wRL_lookup", "parse_surv_wRL_lookup")
        graph.add_edge("init_surv_woRL_lookup", "parse_surv_woRL_lookup")
        graph.add_conditional_edges(
            "parse_surv_wRL_lookup",
            self.make_decision_fun,
            { 1:"init_surv_CL_lookup", 2:"simple_surv_lookup",3:"parse_surv_wRL_lookup"}
        )

        graph.add_conditional_edges(
            "parse_surv_woRL_lookup",
            self.make_decision_fun,
            { 1:"simple_surv_lookup", 2:"init_surv_CL_lookup",3:"parse_surv_woRL_lookup"}
        )

        graph.add_edge("init_surv_CL_lookup", "parse_surv_CL_lookup")

        graph.add_conditional_edges(
            "parse_surv_CL_lookup",
            self.make_decision_fun,
            { 1:"all_surv_lookup", 2:"one_surv_lookup",3:"parse_surv_CL_lookup"}
        )
        
        graph.add_edge("all_surv_lookup",  "init_end")
        graph.add_edge("one_surv_lookup", "init_end")
        graph.add_edge("simple_surv_lookup", "init_end")

        graph.add_conditional_edges(
            "chk_data_lookup",
            self.make_decision_fun,
            {1: "init_VOI_lookup", 2:"init_surv_RL_lookup",3:"chk_data_lookup" }
        )
        graph.add_edge("init_VOI_lookup", "parse_VOI_lookup")
        graph.add_conditional_edges(
            "parse_VOI_lookup",
            self.make_decision_fun,
            {1: "init_RL_lookup", 2:"parse_VOI_lookup" }
        )

        graph.add_conditional_edges(
            "init_RL_lookup",
            self.make_decision_fun,
            {1: "init_wRL_lookup", 2:"init_woRL_lookup" }
        )

        graph.add_edge("init_wRL_lookup", "parse_wRL_lookup")
        graph.add_edge("init_woRL_lookup", "parse_woRL_lookup")
        graph.add_conditional_edges(
            "parse_wRL_lookup",
            self.make_decision_fun,
            { 1:"init_CL_lookup", 2:"simple_CL_lookup",3:"parse_wRL_lookup"}
        )

        graph.add_conditional_edges(
            "parse_woRL_lookup",
            self.make_decision_fun,
            { 1:"simple_CL_lookup", 2:"init_CL_lookup",3:"parse_woRL_lookup"}
        )

        graph.add_edge("init_CL_lookup", "parse_CL_lookup")
        graph.add_conditional_edges(
            "parse_CL_lookup",
            self.make_decision_fun,
            { 1:"all_CL_lookup", 2:"one_CL_lookup",3:"parse_CL_lookup"}
        )
        
        graph.add_edge("all_CL_lookup", "init_end")
        graph.add_edge("one_CL_lookup", "init_end")
        graph.add_edge("simple_CL_lookup", "init_end")
        ###

        graph.add_edge("init_Case", "input_data_Case")

        graph.add_conditional_edges(
            "input_data_Case",
            self.make_decision_fun,
            {1: "load_data_Case", 2:"input_data_Case" }
        )
        
    

        graph.add_edge("load_data_Case", "summary_data_Case")

        graph.add_conditional_edges(
            "summary_data_Case",
            self.make_decision_fun,
            {1: "init_query_I_Case" , 2:"init_Case" ,3:"summary_data_Case"}
        )


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

        graph.add_edge("load_data_Ctrl", "summary_data_Ctrl")

        graph.add_conditional_edges(
            "summary_data_Ctrl",
            self.make_decision_fun,
            {1: "init_query_I_Ctrl" , 2:"init_Ctrl" ,3:"summary_data_Ctrl"}
        )

        
        graph.add_edge("init_query_I_Ctrl", "parse_query_I_Ctrl")

        graph.add_conditional_edges(
            "parse_query_I_Ctrl",
            self.make_decision_fun,
            {1:"overview_Ctrl", 2: "init_set_criteria_Ctrl",  3:"summary_Ctrl" , 4: "init_query_I_Ctrl"}
        )

        graph.add_edge("overview_Ctrl", "show_attr_values_Ctrl")
        graph.add_edge("show_attr_values_Ctrl", "init_query_I_Ctrl")
        
        graph.add_edge("init_set_criteria_Ctrl", "set_criteria_Ctrl")
        graph.add_edge("set_criteria_Ctrl", "init_query_I_Ctrl")
        graph.add_edge("summary_Ctrl", "finalize_data")        
        
        graph.add_conditional_edges(
            "finalize_data",
            self.make_decision_fun,
            {1: "init_exec", 2:"init_Case",3:"finalize_data" }
        )

        
        graph.add_conditional_edges(
            "init_exec",
            self.make_decision_fun,
            {1:"pred_surv", 2:"pred_data"}
        )

        graph.add_edge("pred_surv", "chk_surv")
        graph.add_edge("pred_data", "chk_data")
        
        graph.add_conditional_edges(
            "chk_surv",
            self.make_decision_fun,
            {1: "init_Survival", 2:"init_compare", 3:"chk_surv" }
        )

        graph.add_conditional_edges(
            "chk_data",
            self.make_decision_fun,
            { 1:"init_compare", 2:"init_Survival",3:"chk_data"}
        )
        graph.add_edge("init_compare", "chk_compare")

        graph.add_conditional_edges(
            "chk_compare",
            self.make_decision_fun,
            { 1:"init_OR", 2:"init_CL",3:"chk_compare"}
        )

        graph.add_edge("init_CL", "parse_CL")

        graph.add_conditional_edges(
            "parse_CL",
            self.make_decision_fun,
            { 1:"all_CL", 2:"chk_CL",3:"parse_CL"}
        )

        graph.add_conditional_edges(
            "chk_CL",
            self.make_decision_fun,
            { 1:"one_CL", 2:"init_CL" }
        )
        
        graph.add_edge("all_CL", "init_end")
        graph.add_edge("one_CL", "init_end")

        graph.add_edge("init_OR", "parse_OR")
        graph.add_edge("parse_OR", "init_end")

        graph.add_edge("init_Survival", "parse_Survival")
        graph.add_conditional_edges(
            "parse_Survival",
            self.make_decision_fun,
            { 1:"run_Survival", 2:"init_multiple_Survival",3:"init_Survival"}
        )
        
        graph.add_edge("init_multiple_Survival","multiple_Survival")
        graph.add_edge("multiple_Survival", "run_Survival")
        graph.add_edge("run_Survival", "init_end")    

        graph.add_edge("init_end", "parse_end")
        graph.add_conditional_edges(
            "parse_end",
            self.make_decision_fun,
            { 1:"initQ", 2:"parse_end",3:"parse_end"}
        )

        self.graph = graph.compile(
            checkpointer=local_memory,
            interrupt_before=["init_Qtypechk_1", "pred_twoGroups", "pred_lookup",  "switch_twoGroups","switch_lookup", "input_data_lookup" , "decide_subset_lookup","show_attr_values_lookup", "summary_data_lookup",   "parse_query_I_lookup" , "set_criteria_lookup", "chk_surv_lookup" , "chk_data_lookup", "parse_VOI_lookup", "parse_wRL_lookup", "parse_woRL_lookup",  "parse_surv_wRL_lookup", "parse_surv_woRL_lookup", "parse_CL_lookup", "parse_surv_CL_lookup", "input_data_Case" ,"show_attr_values_Case", "summary_data_Case",  "parse_query_I_Case" , "set_criteria_Case", "input_data_Ctrl" , "summary_data_Ctrl", "parse_query_I_Ctrl" , "show_attr_values_Ctrl","set_criteria_Ctrl","finalize_data", "chk_surv" , "chk_data" , "chk_compare",  "parse_Survival", "multiple_Survival" , "parse_OR", "parse_CL", "parse_end"]
        )
        
        self.conversation_path = ""
        self.model = model
        self.conversation_buffer =[]
        self.main_Q= ""
        self.associated_attr = ""
        self.voi_name =""

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
        self.exhibit_num=1
        self.case_exhibit_num=1
        self.ctrl_exhibit_num=1
        
        self.case_DS_num=1
        self.ctrl_DS_num=1

        self.surv_num=1
        self.surv_extra=[]
        self.CL_num=1

        # self.user_input = tk.StringVar()
        self.html_fname = os.path.normpath("dialogs/welcome.html")


    
    def tk_print(self, input_str):
        try:
            input_str = str(input_str)
        except (ValueError, TypeError):
        
            return

        self.conversation_buffer.append(str(input_str))

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

    def run_script(self, script_fname, arg_fname ):
        command = ['python3', script_fname, arg_fname]
    
        try:
            result = subprocess.run(command, capture_output=False, text=False, check=False)
            print("Script executed successfully.")
            print(result.stdout)  
        except subprocess.CalledProcessError as e:
            print("Error while running script.")
            print(e.stderr)
            return False
        return True


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
    
    def initQ_fun(self, state: AgentState):

        self.main_Q= ""
        
        self.associated_attr = ""
        self.voi_name =""
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

        self.html_fname = os.path.normpath("dialogs/welcome.html")

        with open(os.path.normpath('dialogs/_welcome.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        

    def init_Qtypechk_1_fun(self, state: AgentState):
       
        messages = state['messages'][-1].content
        self.main_Q = messages
        
        with open(os.path.normpath("dialogs/_qtype_1.pkl"), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model
        print("self.main_Q")
        input_dict = {
            "user_input":self.main_Q 
    
        }
        
        output = chain.invoke(
            input_dict
        )

        print(output)

    
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "2"
        
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
                    # self.tk_print("\n[AI] this is a clinical study.\n" )
                    pass
                else:
                    self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                # messages = "Yes"

        return {'messages': [output]}

    
    def init_Qtypechk_2_fun(self, state: AgentState):
        
        with open(os.path.normpath("dialogs/_qtype_2.pkl"), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model  
        input_dict = {
            "user_input":self.main_Q 
        
        }
        output = chain.invoke(
            input_dict
        )
        print(output)

    
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "2"
        
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

        
        return {'messages': [output]}


    def __chk_yn_fun(self, state: AgentState):
        print("__chk_yn_fun")
        messages = state['messages'][-1].content

        with open(os.path.normpath("dialogs/_yn.pkl" ), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model  
        input_dict = {
            "user_input":messages
        
        }
        output = chain.invoke(
            input_dict
        )
        print(output)

    
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "2"
        
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
                    self.tk_print("[AI] you say yes.\n" )
                elif output == "2":
                    self.tk_print("[AI] you say No.\n" )
                else:
                    self.tk_print("[AI] I don't get it. Let's do it again.\n")
                # messages = "Yes"

        return {'messages': [output]}

        print(messages)
        pass

    def make_decision_fun(self, state: AgentState):
        messages = state['messages'][-1].content

        return int(messages)
    
    
    def init_pred_twoGroups_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_pred_twoG.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_switch_twoGroups_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_switch_twoG.pkl') , 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_pred_lookup_fun(self, state: AgentState):
        
        with open(os.path.normpath('dialogs/_init_pred_lookup.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_switch_lookup_fun(self, state: AgentState):

        with open(os.path.normpath('dialogs/_init_switch_lookup.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

    def init_decide_subset_lookup_fun(self, state: AgentState):   
        
        with open(os.path.normpath('dialogs/_init_decide_subset_lookup.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_lookup_fun(self, state: AgentState):
        
        with open(os.path.normpath('dialogs/_init_lookup.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

        self.tk_print("The following datasets have been successfully installed in your computer:\n")
        df = pd.read_csv(os.path.normpath('data/dataset.tsv'), sep='\t' ,na_values=["none", ""])
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

    def init_VOI_lookup_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_VOI_lookup.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str) 
        str=f"[AI] You can check the attribute names in your data table. Your data table is located at {self.Case_metafname}.\n"
        self.tk_print(str)

    def parse_VOI_lookup_fun(self, state: AgentState):
        output = ""
        messages = state['messages'][-1].content
        output = self.find_best_match(messages, self.Case_metadata_df.columns  )
        
        messages = "2"
        
        if  output==""  :
                self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        else :
            self.voi_name = output
            self.tk_print(f"\n[AI] You selected {self.voi_name}.\n")
            messages = "1"

        return {'messages': [messages]}

    def init_Case_fun(self, state: AgentState):
        self.html_fname = os.path.normpath("dialogs/cc.html")
        with open(os.path.normpath('dialogs/init_case.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

        with open(os.path.normpath('dialogs/init_case_1.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

        self.tk_print("The following datasets have been successfully installed in your computer:\n")
        df = pd.read_csv(os.path.normpath('data/dataset.tsv'), sep='\t' ,na_values=["none", ""])
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
     
    def load_data_Case_fun(self, state: AgentState):
        index_fname = os.path.normpath("data/{}/INDEX.tsv".format(self.Case_data_id ) )
        df = pd.read_csv(index_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])

        config_dict = {}
        
        for key in df.index:
            config_dict[key] = df.loc[ key, "value" ] 
        
        metadata_fname = os.path.normpath("data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] ) )
        self.Case_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
        self.Case_config_dict = config_dict
        self.Case_metadata_df = df
       
        self.Case_metadata_df = self.Case_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
       
        rows, columns = self.Case_metadata_df.shape


        str ="\n=======================================================\n"+ "Introduction to the Dataset" +"\n=======================================================\n"
        self.tk_print(str)
        index_fname = os.path.normpath( "data/{}/{}".format(self.Case_data_id, self.Case_config_dict["README"]) )
        with open(index_fname, "r") as f:
            file_content = f.read()
        f.close()
        self.tk_print(file_content)

        str=f"[AI] Your data table is located at {metadata_fname}.\n"
        self.tk_print(str)
        str=f"[AI] There are {rows} samples and {columns} attributes in your dataset.\n"
        self.tk_print(str)
        str ="\n=======================================================\n"+ "Confirm Selected Data" +"\n=======================================================\n"
        self.tk_print(str)
        str=f"\nIs this the correct dataset you'd like to use for your analysis?\nPlease reply YES or NO."
        self.tk_print(str)
        


    
    def init_set_criteria_fun(self, state: AgentState):
        with open(os.path.normpath( 'dialogs/set_up_criteria.pkl') , 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        
        pass

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
            #####
            # input_string = self.extract_relationship_prompt(condition)

            with open(os.path.normpath("dialogs/_relationship.pkl"), "rb") as f:
                loaded_chat_prompt = pickle.load(f)

            chain = loaded_chat_prompt| self.model  
            input_dict = {
                "user_input":condition
        
            }
            input_string = chain.invoke(
                input_dict
            )
            #####
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
        out_html_fname = os.path.normpath(self.conversation_path+"/case_sample_selection.html")
        self.Case_criteria_str = new_expression
        self.Case_criteria_logic = condition_map
        msg_dict ={
        "case_id":"Case",
        "total_num":self.Case_metadata_df.shape[0],
        "criteria_str":self.Case_criteria_str ,
        "criteria_logic":self.Case_criteria_logic ,
        "selected_num":len(self.Case_sample_ids),
        "output_path":self.conversation_path,
        "output_png":os.path.normpath(self.conversation_path+"/case_sample_selection_"+str(self.case_DS_num)+".png" ),
        "output_html":os.path.normpath( self.conversation_path+"/case_sample_selection_"+str(self.case_DS_num)+".html" )
        }

        with open( os.path.normpath(self.conversation_path+'/case_sample_selection+'+str(self.case_DS_num)+'.pkl' ), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        
        time.sleep(1)
        if self.run_script( "DS_Agent.py",os.path.normpath(self.conversation_path+'/case_sample_selection+'+str(self.case_DS_num)+'.pkl' ) ):
            self.html_fname = msg_dict["output_html"]
            self.tk_print(f"[AI] Congratulations! You have successfully set up the criteria to refine the samples. You can now proceed to the next step.\n" )
            self.case_DS_num = self.case_DS_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        return {'messages': [output]}  

    
    def overview_Case_fun(self, state: AgentState): 
        
        self.tk_print("[AI] Please enter the name of a data attribute, and I can display the distribution of its values.")
        str=f"[AI] You can check the attribute names in your data table. Your data table is located at {self.Case_metafname}.\n"
        self.tk_print(str)

    def show_attr_values_Case_fun(self, state: AgentState):

        # self.tk_print("show_attr_values_Case_fun")
        messages = state['messages'][-1].content
        data_attr = self.find_best_match(messages, self.Case_metadata_df.columns  )
        if data_attr != "" :
            self.tk_print(data_attr)

            msg_dict ={
            "metafname":self.Case_metafname,    
            "Attr_ID":data_attr,
            "output_path":self.conversation_path,
            "output_png":os.path.normpath(self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".png" ),
            "output_html":os.path.normpath(self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".html" )
            }
        
            with open( os.path.normpath(self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".pkl" ), 'wb') as f:
                pickle.dump(msg_dict, f)
            f.close()
            time.sleep(1)
            if self.run_script( "EXHIBIT_Agent.py",os.path.normpath(self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".pkl" ) ):
                self.html_fname = os.path.normpath(self.conversation_path+"/Case_EXHIBIT_"+str(self.case_exhibit_num) +".html" )
                self.case_exhibit_num = self.case_exhibit_num+1
            else:
                
                self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

    def summary_Case_fun(self, state: AgentState): 
        self.html_fname = os.path.normpath("dialogs/cc_2.html" )
        if len(self.Case_criteria_logic) ==0:
            self.Case_sample_ids = self.Case_metadata_df.index.to_list()
            str =f"[AI] You have not defined any criteria to filter samples for the case cohort. All {len(self.Case_sample_ids)} samples in the dataset will be included."
            self.tk_print(str)
           
        else:
            str =f"[AI] You have defined {len(self.Case_criteria_logic)} criteria and selected {len(self.Case_sample_ids)} samples for the case cohort."
            self.tk_print(str)

    def init_Ctrl_fun(self, state: AgentState):
        with open(os.path.normpath( 'dialogs/init_ctrl_1.pkl' ) , 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        
        
        
        self.tk_print("The following datasets have been successfully installed in your computer:\n")
        df = pd.read_csv(os.path.normpath('data/dataset.tsv'), sep='\t' ,na_values=["none", ""])
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
   
    def init_query_I_Case_fun(self, state: AgentState):

        with open(os.path.normpath('dialogs/parse_query_I.pkl' ), 'rb') as f:
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

        with open(os.path.normpath('dialogs/parse_query_I.pkl' ), 'rb') as f:
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
        
        with open(os.path.normpath("dialogs/_query_I.pkl" ), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model  
        input_dict = {
            "user_input":messages
        
        }
        output = chain.invoke(
            input_dict
        )
        
        
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
                    self.tk_print("\n[AI] You want to proceed to the next step.\n" )
                else:
                    self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
                # messages = "Yes"

        return {'messages': [output]}
        
    

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
        index_fname = os.path.normpath("data/{}/INDEX.tsv".format(self.Ctrl_data_id ) )
        df = pd.read_csv(index_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])

        config_dict = {}
        
        for key in df.index:
            config_dict[key] = df.loc[ key,  "value"  ]  
        metadata_fname = os.path.normpath("data/{}/{}".format(self.Case_data_id,config_dict["DATAFNAME"] ) )
        self.Ctrl_metafname = metadata_fname
        df = pd.read_csv(metadata_fname, sep="\t", index_col=0,  header=0 ,na_values=["none", ""])
    
        self.Ctrl_metadata_df = df
        self.Ctrl_config_dict = config_dict
        self.Ctrl_metadata_df = self.Ctrl_metadata_df.apply(lambda col: col.astype('string') if col.dtype == 'object' else col)
        rows, columns = self.Ctrl_metadata_df.shape
        str ="\n=======================================================\n"+ "Introduction to the Dataset" +"\n=======================================================\n"
        self.tk_print(str)
        index_fname = os.path.normpath( "data/{}/{}".format(self.Ctrl_data_id, self.Ctrl_config_dict["README"]) )
        with open(index_fname, "r") as f:
            file_content = f.read()
        f.close()
        self.tk_print(file_content)

        str=f"[AI] Your data table is located at {metadata_fname}.\n"
        self.tk_print(str)
        str=f"[AI] There are {rows} samples and {columns} attributes in your dataset.\n"
        self.tk_print(str)
        str ="\n=======================================================\n"+ "Confirm Selected Data" +"\n=======================================================\n"
        self.tk_print(str)
        str=f"\nIs this the correct dataset you'd like to use for your analysis?\nPlease reply YES or NO."
        self.tk_print(str)

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
            #####
            with open(os.path.normpath( "dialogs/_relationship.pkl" ), "rb") as f:
                loaded_chat_prompt = pickle.load(f)

            chain = loaded_chat_prompt| self.model  
            input_dict = {
                "user_input":condition
        
            }
            input_string = chain.invoke(
                input_dict
            )
            # input_string = self.extract_relationship_prompt(condition)
            #####
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
        out_html_fname = os.path.normpath( self.conversation_path+"/ctrl_sample_selection.html" )
        self.Ctrl_criteria_str = new_expression
        self.Ctrl_criteria_logic = condition_map
        msg_dict ={
        "case_id":"Control",
        "total_num":self.Ctrl_metadata_df.shape[0],
        "criteria_str":self.Ctrl_criteria_str ,
        "criteria_logic":self.Ctrl_criteria_logic ,
        "selected_num":len(self.Ctrl_sample_ids),
        "output_path":self.conversation_path,
        "output_png":os.path.normpath( self.conversation_path+"/ctrl_sample_selection_"+str(self.ctrl_DS_num)+".png" ),
        "output_html":os.path.normpath( self.conversation_path+"/ctrl_sample_selection_"+str(self.ctrl_DS_num)+".html" )
        }

        with open( os.path.normpath(self.conversation_path+'/ctrl_sample_selection+'+str(self.ctrl_DS_num)+'.pkl') , 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()

      
        time.sleep(1)
        if self.run_script( "DS_Agent.py",os.path.normpath(self.conversation_path+'/ctrl_sample_selection+'+str(self.ctrl_DS_num)+'.pkl' ) ):
            self.html_fname = msg_dict["output_html"]
            self.ctrl_DS_num = self.ctrl_DS_num+1
            self.tk_print(f"[AI] Congratulations! You have successfully set up the criteria to refine the control samples. You can now proceed to the next step.\n" )
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        return {'messages': [output]}  
  
    def overview_Ctrl_fun(self, state: AgentState): 
        
        self.tk_print("[AI] Please enter the name of a data attribute, and I can display the distribution of its values.")
        str=f"[AI] You can check the attribute names in your data table. Your data table is located at {self.Ctrl_metafname}.\n"
        self.tk_print(str)

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
            "output_png":os.path.normpath(self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".png" ),
            "output_html":os.path.normpath(self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".html")
            }
        
            with open( os.path.normpath(self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".pkl" ), 'wb') as f:
                pickle.dump(msg_dict, f)
            f.close()
            time.sleep(1)
            if self.run_script( "EXHIBIT_Agent.py",os.path.normpath(self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".pkl" ) ):
                self.html_fname = os.path.normpath(self.conversation_path+"/Ctrl_EXHIBIT_"+str(self.ctrl_exhibit_num) +".html")
                self.ctrl_exhibit_num = self.ctrl_exhibit_num+1
            else:
                self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")

    
    def summary_Ctrl_fun(self, state: AgentState): 
        self.html_fname = os.path.normpath("dialogs/cc_3.html")
        str ="\n=======================================================\n"+ "Finalize Your Data Selection for the Case and Control Groups" +"\n=======================================================\n"
        self.tk_print(str)

        if len(self.Case_criteria_logic) ==0:
            self.Ctrl_sample_ids = self.Ctrl_metadata_df.index.to_list()
            str =f"[AI] You have not defined any criteria to filter samples for the control cohort. All {len(self.Ctrl_sample_ids)} samples in the dataset will be included."
            self.tk_print(str)
        else:
            str =f"[AI] You have defined {len(self.Ctrl_criteria_logic)} criteria and selected {len(self.Ctrl_sample_ids)} samples for the contrl cohort."
            self.tk_print(str)
        
        str =f"[AI] You have also defined {len(self.Case_criteria_logic)} criteria and selected {len(self.Case_sample_ids)} samples for the case cohort."
        self.tk_print(str)
        output = "1"
        
        if self.Ctrl_data_id == self.Ctrl_data_id :
            if set(self.Ctrl_sample_ids) & set(self.Case_sample_ids) :
                
                str =f"[AI] *** Warning *** There are {len(set(self.Ctrl_sample_ids) & set(self.Case_sample_ids))} samples shared in the case and control cohorts. Please revise the sample selection."
                self.tk_print(str)
                output = "2"
        
        
        str =f"\n[AI] You have now set up your case and control cohorts. Please confirm to proceed. (Yes or No)"
        self.tk_print(str)

        return {'messages': [output]}


    
    
    def init_exec_fun(self, state: AgentState):
       
        
        with open(os.path.normpath("dialogs/_surv.pkl"), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model  
        input_dict = {
            "user_input":self.main_Q 
        
        }
        output = chain.invoke(
            input_dict
        )
        print(output)

    
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "2"
        
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
                    self.tk_print("[AI] this is a surv control study.\n" )

                else:
                    self.tk_print("[AI] this is not a surv control study.\n")
                # messages = "Yes"

        
        return {'messages': [output]}

    def init_OR_fun(self, state: AgentState):   
        with open(os.path.normpath('dialogs/init_OR.pkl'), 'rb') as f:
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
            ####
            #input_string = self.extract_relationship_prompt(condition)
            with open(os.path.normpath("dialogs/_relationship.pkl"), "rb") as f:
                loaded_chat_prompt = pickle.load(f)

            chain = loaded_chat_prompt| self.model  
            input_dict = {
                "user_input":condition
        
            }
            input_string = chain.invoke(
                input_dict
            )
            ###
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
        "output_png":os.path.normpath(self.conversation_path+"/OR_test_"+str(self.or_num) +".png" ),
        "output_html":os.path.normpath(self.conversation_path+"/OR_test_"+str(self.or_num) +".html")
        }

        with open( os.path.normpath( self.conversation_path+"/OR_test_"+str(self.or_num) +".pkl" ), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "OR_Agent.py",os.path.normpath(self.conversation_path+"/OR_test_"+str(self.or_num) +".pkl" ) ):
            self.html_fname = os.path.normpath(self.conversation_path+"/OR_test_"+str(self.or_num) +".html" )
            self.or_num = self.or_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        return {'messages': [output]}  
    
    def init_compare_fun(self, state: AgentState):
        with open(os.path.normpath( 'dialogs/_init_compare.pkl' ), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)    

    def init_Survival_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/init_Survival.pkl' ), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  
        pass
      
    
    def init_multiple_Survival_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/multi_Survival.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  
   

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
        "output_OS_png":os.path.normpath(self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png" ),
        "output_PFS_png":os.path.normpath(self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png"),
        "output_forest_OS_png":os.path.normpath(self.conversation_path+"/Surv_forest_OS_"+str(self.surv_num) +".png"),
        "output_forest_PFS_png":os.path.normpath(self.conversation_path+"/Surv_forest_PFS_"+str(self.surv_num) +".png"),
        "output_html":os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".html")
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
            msg_dict["output_OS_png"]=os.path.normpath(self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png" )
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
            msg_dict["output_PFS_png"]=os.path.normpath(self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png")
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 
            msg_dict["Ctrl_PFS_TIME"] = self.Ctrl_config_dict["PFS_TIME"] 
            msg_dict["Ctrl_PFS_STATUS"] = self.Ctrl_config_dict["PFS_STATUS"] 
        

        
        msg_dict["EXTRA_ATTR"] = self.surv_extra
        

        with open( os.path.normpath( self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" ), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "SURV_Agent.py",os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" ) ):
            self.html_fname = os.path.normpath( self.conversation_path+"/Surv_"+str(self.surv_num) +".html" )
            self.surv_extra = []
            self.surv_num = self.surv_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        
    
    def pred_surv_fun(self, state: AgentState):


        with open(os.path.normpath( 'dialogs/_pred_surv.pkl' ), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        self.tk_print(f"Your initial input is : {self.main_Q}\n")
        str = loaded_dict["message"]
        self.tk_print(str)  

    def pred_data_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_pred_data.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        self.tk_print(f"Your initial input is : {self.main_Q}\n")
        str = loaded_dict["message"]
        self.tk_print(str)  
    
    def init_wRL_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_wRL.pkl' ), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_woRL_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_woRL.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  

    def init_RL_fun(self, state: AgentState):

        with open(os.path.normpath("dialogs/_relation_chk.pkl"), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model
        input_dict = {
            "user_input":self.main_Q 
        
        }
        output = chain.invoke(
            input_dict
        )
    
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "2"
        
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

        return {'messages': [output]}

    def init_CL_fun(self, state: AgentState):
        with open(os.path.normpath('dialogs/_init_CL.pkl' ), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)  
    
    def init_end_fun(self, state: AgentState):
        
        with open(os.path.normpath('dialogs/_init_END.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        self.tk_print(self.conversation_path)
        self.tk_print("\n[AI] Would you like to perform another analysis? Please reply yes to continue")
      
  
    
    def parse_CL_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        
        with open(os.path.normpath("dialogs/_all.pkl"), "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt| self.model  
        input_dict = {
            "user_input":messages
        
        }
        output = chain.invoke(
            input_dict
        )
        
        
        matches = re.findall(r'\[(.*?)\]', output)
        
        data_id_list = []
        output = "3"
        
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
                if output == "2":
                    self.tk_print("\n[AI] You want to one variable.\n" )
                    attribute_id  = self.find_best_match( messages, self.Case_metadata_df.columns)
                    self.associated_attr = attribute_id
                    self.tk_print(f"[AI] {self.associated_attr} is used here.\n")


            return {'messages': [output]}  
    def chk_CL_fun(self, state: AgentState):
        output = "2"
        if self.associated_attr in self.Ctrl_metadata_df.columns :
            output = "1"    
        else:
            self.tk_print(f"[AI]***WARNING*** {self.associated_attr} is not defined in the control samples.\n")
        return {'messages': [output]}  
    def one_CL_fun(self, state: AgentState):
        
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "Ctrl_metafname":self.Ctrl_metafname,
        "Ctrl_ID":self.Ctrl_sample_ids,
        "Case_criteria_str":self.Case_criteria_str,
        "Ctrl_criteria_str":self.Ctrl_criteria_str,
        "Case_criteria_logic":self.Case_criteria_logic,
        "Ctrl_criteria_logic":self.Ctrl_criteria_logic,
        "associated_attr":self.associated_attr ,
        "output_path":self.conversation_path,
        "output_png":os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".png"),
        "output_html":os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
        }
        with open( os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" ), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "One_v_Groups.py",os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" ) ):
            self.html_fname = os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html" )
            self.CL_num = self.CL_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")

    def all_CL_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "Ctrl_metafname":self.Ctrl_metafname,
        "Ctrl_ID":self.Ctrl_sample_ids,
        "Case_criteria_str":self.Case_criteria_str,
        "Ctrl_criteria_str":self.Ctrl_criteria_str,
        "Case_criteria_logic":self.Case_criteria_logic,
        "Ctrl_criteria_logic":self.Ctrl_criteria_logic,
        "output_path":self.conversation_path,
        "output_tsv":os.path.normpath(self.conversation_path+"/ALL_vs_Groups_"+str(self.CL_num) +".tsv" ),
        "output_html":os.path.normpath( self.conversation_path+"/CL_"+str(self.CL_num) +".html" )
        }
        with open( os.path.normpath( self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" ), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "All_v_Groups.py",os.path.normpath( self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" )):
            self.html_fname = os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
            self.CL_num = self.CL_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")

    def one_CL_lookup_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "voi_name":self.voi_name,
        "associated_attr":self.associated_attr ,
        "output_path":self.conversation_path,
        "output_png":os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".png"),
        "output_html":os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
        }

        with open( os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" ) , 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "One_v_One_CL.py",os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl" )) :
            self.html_fname = os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
            self.CL_num = self.CL_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
    
    def simple_CL_lookup_fun(self, state: AgentState):
        
        if self.voi_name != "" :

            msg_dict ={
            "metafname":self.Case_metafname,    
            "Attr_ID":self.voi_name,
            "output_path":self.conversation_path,
            "output_png":os.path.normpath(self.conversation_path+"/EXHIBIT_"+str(self.exhibit_num) +".png"),
            "output_html":os.path.normpath(self.conversation_path+"/EXHIBIT_"+str(self.exhibit_num) +".html")
            }
        
            with open( os.path.normpath(self.conversation_path+"/EXHIBIT_"+str(self.exhibit_num) +".pkl"), 'wb') as f:
                pickle.dump(msg_dict, f)
            f.close()
            time.sleep(1)
            if self.run_script( "EXHIBIT_Agent.py",os.path.normpath(self.conversation_path+"/EXHIBIT_"+str(self.exhibit_num) +".pkl") ):
                self.html_fname = os.path.normpath(self.conversation_path+"/EXHIBIT_"+str(self.exhibit_num) +".html")
                self.exhibit_num = self.exhibit_num+1
            else:
                
                self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")



    def all_CL_lookup_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "voi_name":self.voi_name,
        "output_path":self.conversation_path,
        "output_tsv":os.path.normpath(self.conversation_path+"/"+self.voi_name+"_vs_ALL_"+str(self.CL_num) +".tsv"),
        "output_html":os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
        }

        with open( os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl"), 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "All_v_One_CL.py",os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".pkl") ) :
            self.html_fname = os.path.normpath(self.conversation_path+"/CL_"+str(self.CL_num) +".html")
            self.CL_num = self.CL_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")

    def one_surv_lookup_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "associated_attr":self.associated_attr ,
        "output_path":self.conversation_path,
        "output_html":os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".html")
        }
        
        OS_flag =1 
        if "OS_TIME" in self.Case_config_dict and self.Case_config_dict["OS_TIME"].strip() !="":
            pass
        else:
            OS_flag=0
        
        if "OS_STATUS" in self.Case_config_dict and self.Case_config_dict["OS_STATUS"].strip() !="":
            pass
        else:
            OS_flag=0

        
        if OS_flag ==1:
            msg_dict["output_KM_OS_png"]=os.path.normpath(self.conversation_path+"/Surv_OS_"+str(self.surv_num) +"_KM.png")
            msg_dict["output_forest_OS_png"]=os.path.normpath(self.conversation_path+"/Surv_OS_"+str(self.surv_num) +"_forest.png")
            msg_dict["Case_OS_TIME"] = self.Case_config_dict["OS_TIME"] 
            msg_dict["Case_OS_STATUS"] = self.Case_config_dict["OS_STATUS"] 


        
        PFS_flag =1 
        if "PFS_TIME" in self.Case_config_dict and self.Case_config_dict["PFS_TIME"].strip() !="":
            pass
        else:
            PFS_flag=0
        
        if "PFS_STATUS" in self.Case_config_dict and self.Case_config_dict["PFS_STATUS"].strip() !="":
            pass
        else:
            PFS_flag=0

        
        if PFS_flag ==1:
            msg_dict["output_KM_PFS_png"]=os.path.normpath(self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +"_KM.png")
            msg_dict["output_forest_PFS_png"]=os.path.normpath( self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +"_forest.png" )
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 

        
        with open( os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl") , 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "one_SURV_lookup.py",self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" ):
            self.html_fname = os.path.normpath( self.conversation_path+"/Surv_"+str(self.surv_num) +".html")
            self.surv_num = self.surv_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")
    
    def simple_surv_lookup_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "output_path":self.conversation_path,
        "output_html":os.path.normpath( self.conversation_path+"/Surv_"+str(self.surv_num) +".html" )
        }
        
        OS_flag =1 
        if "OS_TIME" in self.Case_config_dict and self.Case_config_dict["OS_TIME"].strip() !="":
            pass
        else:
            OS_flag=0
        
        if "OS_STATUS" in self.Case_config_dict and self.Case_config_dict["OS_STATUS"].strip() !="":
            pass
        else:
            OS_flag=0

        
        if OS_flag ==1:
            msg_dict["output_KM_OS_png"]=os.path.normpath( self.conversation_path+"/Surv_OS_"+str(self.surv_num) +"_KM.png")
            msg_dict["Case_OS_TIME"] = self.Case_config_dict["OS_TIME"] 
            msg_dict["Case_OS_STATUS"] = self.Case_config_dict["OS_STATUS"] 


        
        PFS_flag =1 
        if "PFS_TIME" in self.Case_config_dict and self.Case_config_dict["PFS_TIME"].strip() !="":
            pass
        else:
            PFS_flag=0
        
        if "PFS_STATUS" in self.Case_config_dict and self.Case_config_dict["PFS_STATUS"].strip() !="":
            pass
        else:
            PFS_flag=0

        
        if PFS_flag ==1:
            msg_dict["output_KM_PFS_png"]=os.path.normpath(self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +"_KM.png" )
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 

        
        with open( os.path.normpath( self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" ) , 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)
        if self.run_script( "one_SURV_lookup.py",os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" )):
            self.html_fname = os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".html")
            self.surv_num = self.surv_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")


    def all_surv_lookup_fun(self, state: AgentState):
        msg_dict ={
        "Case_metafname":self.Case_metafname,    
        "Case_ID":self.Case_sample_ids,
        "output_path":self.conversation_path,
        "output_html":os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".html"),
        "output_tsv":os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".tsv")
        }
        
        OS_flag =1 
        if "OS_TIME" in self.Case_config_dict and self.Case_config_dict["OS_TIME"].strip() !="":
            pass
        else:
            OS_flag=0
        
        if "OS_STATUS" in self.Case_config_dict and self.Case_config_dict["OS_STATUS"].strip() !="":
            pass
        else:
            OS_flag=0

        
        if OS_flag ==1:
            msg_dict["output_OS_png"]=os.path.normpath(self.conversation_path+"/Surv_OS_"+str(self.surv_num) +".png")
            msg_dict["Case_OS_TIME"] = self.Case_config_dict["OS_TIME"] 
            msg_dict["Case_OS_STATUS"] = self.Case_config_dict["OS_STATUS"] 


        
        PFS_flag =1 
        if "PFS_TIME" in self.Case_config_dict and self.Case_config_dict["PFS_TIME"].strip() !="":
            pass
        else:
            PFS_flag=0
        
        if "PFS_STATUS" in self.Case_config_dict and self.Case_config_dict["PFS_STATUS"].strip() !="":
            pass
        else:
            PFS_flag=0

        
        if PFS_flag ==1:
            msg_dict["output_PFS_png"]=os.path.normpath(self.conversation_path+"/Surv_PFS_"+str(self.surv_num) +".png")
            msg_dict["Case_PFS_TIME"] = self.Case_config_dict["PFS_TIME"] 
            msg_dict["Case_PFS_STATUS"] = self.Case_config_dict["PFS_STATUS"] 

        
        with open(os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl" ) , 'wb') as f:
            pickle.dump(msg_dict, f)
        f.close()
        time.sleep(1)

        if self.run_script( "All_SURV_lookup.py",os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".pkl") ):
            self.html_fname = os.path.normpath(self.conversation_path+"/Surv_"+str(self.surv_num) +".html" )
            self.surv_num = self.surv_num+1
        else:
            self.tk_print("[AI] *** Warning *** An error occurred while running the data analysis agents.")


    def pop_messages(self):
        if(len(self.conversation_buffer) ==0):
            return ""

        output = "\n".join(self.conversation_buffer)
        self.conversation_buffer=[] 
        return output

    def start(self,thread, thread_id , conversation_path):
         
        self.conversation_path = os.path.normpath(conversation_path) 
        if not os.path.exists(self.conversation_path):
            os.makedirs(self.conversation_path)
        
        for event in self.graph.stream({"messages": ("user", "")} ,thread):
            for value in event.values():
                print(value)
        snapshot = self.graph.get_state(thread)
        print(snapshot)

    def proceed(self,thread, thread_id, input_str):

        snapshot = self.graph.get_state(thread)
        # print(snapshot)
        
        if snapshot.next[0]=="init_Qtypechk_1" :
            node_str =  "initQ"
        if snapshot.next[0]=="pred_twoGroups" :
            node_str =  "init_pred_twoGroups"
        if snapshot.next[0]=="pred_lookup" :
            node_str =  "init_pred_lookup"
        if snapshot.next[0]=="switch_twoGroups" :
            node_str =  "init_switch_twoGroups"
        if snapshot.next[0]=="switch_lookup" :
            node_str =  "init_switch_lookup"
        if snapshot.next[0]=="decide_subset_lookup" :
            node_str =  "init_decide_subset_lookup"
        
        if snapshot.next[0]=="input_data_lookup" :
            node_str =  "init_lookup"
        if snapshot.next[0]=="summary_data_lookup" :
            node_str =  "load_data_lookup"
        if snapshot.next[0]=="parse_query_I_lookup" :
            node_str =  "init_query_I_lookup"
        if snapshot.next[0]=="set_criteria_lookup" :
            node_str =  "init_set_criteria_lookup"
        if snapshot.next[0]=="show_attr_values_lookup" :
            node_str = "overview_lookup"
        if snapshot.next[0]=="chk_surv_lookup" :
            node_str =  "pred_surv_lookup"
        if snapshot.next[0]=="chk_data_lookup" :
            node_str =  "pred_data_lookup"
        
        if snapshot.next[0]=="parse_VOI_lookup" :
            node_str =  "init_VOI_lookup"
        if snapshot.next[0]=="parse_CL_lookup" :
            node_str =  "init_CL_lookup"
        if snapshot.next[0]=="parse_surv_CL_lookup" :
            node_str =  "init_surv_CL_lookup"
        if snapshot.next[0]=="parse_wRL_lookup" :
            node_str =  "init_wRL_lookup"
        if snapshot.next[0]=="parse_woRL_lookup" :
            node_str =  "init_woRL_lookup"
        if snapshot.next[0]=="parse_surv_wRL_lookup" :
            node_str =  "init_surv_wRL_lookup"
        if snapshot.next[0]=="parse_surv_woRL_lookup" :
            node_str =  "init_surv_woRL_lookup"
        

        if snapshot.next[0]=="input_data_Case" :
            node_str =  "init_Case"
        if snapshot.next[0]=="summary_data_Case" :
            node_str =  "load_data_Case"
        if snapshot.next[0]=="parse_query_I_Case" :
            node_str =  "init_query_I_Case"
        if snapshot.next[0]=="set_criteria_Case" :
            node_str =  "init_set_criteria_Case"
        if snapshot.next[0]=="show_attr_values_Case" :
            node_str = "overview_Case"

        if snapshot.next[0]=="input_data_Ctrl" :
            node_str =  "init_Ctrl"
        if snapshot.next[0]=="summary_data_Ctrl" :
            node_str =  "load_data_Ctrl"
        if snapshot.next[0]=="parse_query_I_Ctrl" :
            node_str =  "init_query_I_Ctrl"
        if snapshot.next[0]=="set_criteria_Ctrl" :
            node_str =  "init_set_criteria_Ctrl"
        if snapshot.next[0]=="show_attr_values_Ctrl" :
            node_str = "overview_Ctrl"
        
        if snapshot.next[0]=="finalize_data" :
            node_str =  "summary_Ctrl"
            
        if snapshot.next[0]=="chk_surv" :
            node_str =  "pred_surv"
        if snapshot.next[0]=="chk_data" :
            node_str =  "pred_data"

        if snapshot.next[0]=="parse_OR" :
            node_str = "init_OR"
        
        if snapshot.next[0]=="parse_Survival" :
            node_str = "init_Survival"
        if snapshot.next[0]=="multiple_Survival" :
            node_str = "init_multiple_Survival"
        if snapshot.next[0]=="chk_compare" :
            node_str = "init_compare"
        if snapshot.next[0]=="parse_CL" :
            node_str = "init_CL"
        if snapshot.next[0]=="parse_end" :
            node_str = "init_end"

        self.graph.update_state(
                    thread,
                    {"messages": [input_str]},
                    as_node= node_str
        )
        for event in self.graph.stream(None ,thread):
                for value in event.values():
                    print(value)
        snapshot = self.graph.get_state(thread)
        if len(snapshot.next)==0 :
            return False
        
        return True
        

def main():
    user_input = ""

    # Define the LLM
    llm =  OllamaLLM(model="llama3",temperature=0)

    # Thread
    thread_p1 = {"configurable": {"thread_id": "1"}}
    memory_p1 = MemorySaver()

    abot = Boss( llm, memory_p1  )
    
    abot.start(thread_p1, "1" , "conversation/test/")
        
    while True :
        conversation_content =abot.pop_messages()
        print(conversation_content)
            # conversation_content = self.user_input.get()
            # self.output_text.config(state=tk.NORMAL)
            # self.output_text.insert(tk.END, conversation_content+"\n")
            # self.output_text.config(state=tk.DISABLED)
            # self.output_text.see(tk.END)
            
            # self.display_html(abot.html_fname)
            # print(abot.html_fname)
            # self.user_input.set('')   
            # self.root.wait_variable(self.user_input)

        input_str = input()
            
        if input_str.lower() in ["quit", "exit", "q"]:
            break
        if ( not abot.proceed(thread_p1, "1",input_str) ) :
            break
        # self.root.quit() 
        # self.root.destroy() 
    print("Goodbye!")
        # print("All the statistical reports are generated at " + self.conversation_path  + ".")
     

    # from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
    # from PIL import Image
    # from io import BytesIO

    
    # image_stream = BytesIO(abot.graph.get_graph().draw_png())

    # # Open the image using PIL
    # image = Image.open(image_stream)

    # image.save("saved_image.png")

if __name__=="__main__":
    

    main()
