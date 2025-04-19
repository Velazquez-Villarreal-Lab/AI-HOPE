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

class Data_manager:
    def __init__(self,  model, local_memory ):
        graph = StateGraph(AgentState)
        
        
        graph.add_node("initQ", self.initQ_fun)
        graph.add_node("init_Qtypechk_1", self.init_Qtypechk_1_fun)
        
        graph.add_node("init_create", self.init_create_fun)

        graph.add_node("init_DataFname", self.init_DataFname_fun)
        graph.add_node("parse_DataFname", self.parse_DataFname_fun)
        graph.add_node("init_chk_DataFname", self.init_chk_DataFname_fun)
        graph.add_node("chk_DataFname", self.__chk_yn_fun)


        graph.add_node("init_DataID", self.init_DataID_fun)
        graph.add_node("parse_DataID", self.parse_DataID_fun)
        graph.add_node("init_chk_DataID", self.init_chk_DataID_fun)
        graph.add_node("chk_DataID", self.__chk_yn_fun)

        graph.add_node("init_DataDS", self.init_DataDS_fun)
        graph.add_node("parse_DataDS", self.parse_DataDS_fun)
        graph.add_node("init_chk_DataDS", self.init_chk_DataDS_fun)
        graph.add_node("chk_DataDS", self.__chk_yn_fun)

        graph.add_node("init_DataREADME", self.init_DataREADME_fun)
        graph.add_node("parse_DataREADME", self.parse_DataREADME_fun)
        graph.add_node("init_chk_DataREADME", self.init_chk_DataREADME_fun)
        graph.add_node("chk_DataREADME", self.__chk_yn_fun)

        graph.add_node("init_OS_Time", self.init_OS_Time_fun)
        graph.add_node("parse_OS_Time", self.parse_OS_Time_fun)
        graph.add_node("init_chk_OS_Time", self.init_chk_OS_Time_fun)
        graph.add_node("chk_OS_Time", self.__chk_yn_fun)

        graph.add_node("init_OS_Events", self.init_OS_Events_fun)
        graph.add_node("parse_OS_Events", self.parse_OS_Events_fun)
        graph.add_node("init_chk_OS_Events", self.init_chk_OS_Events_fun)
        graph.add_node("chk_OS_Events", self.__chk_yn_fun)
        
        graph.add_node("init_PFS_Time", self.init_PFS_Time_fun)
        graph.add_node("parse_PFS_Time", self.parse_PFS_Time_fun)
        graph.add_node("init_chk_PFS_Time", self.init_chk_PFS_Time_fun)
        graph.add_node("chk_PFS_Time", self.__chk_yn_fun)

        graph.add_node("init_PFS_Events", self.init_PFS_Events_fun)
        graph.add_node("parse_PFS_Events", self.parse_PFS_Events_fun)
        graph.add_node("init_chk_PFS_Events", self.init_chk_PFS_Events_fun)
        graph.add_node("chk_PFS_Events", self.__chk_yn_fun)
        
        graph.add_node("create_newdata", self.create_newdata_fun)

        graph.add_node("init_delete", self.init_delete_fun)
        graph.add_node("parse_delete", self.parse_delete_fun)
        graph.add_node("init_chk_delete", self.init_chk_delete_fun)
        graph.add_node("chk_delete", self.__chk_yn_fun)
        graph.add_node("delete_data", self.delete_data_fun)

        graph.add_edge(START, "initQ")
        graph.add_edge("initQ", "init_Qtypechk_1")
        graph.add_conditional_edges(
            "init_Qtypechk_1",
            self.make_decision_fun,
            {1: "init_create", 2:"init_delete",  3:"initQ" }
        )
        graph.add_edge("init_create", "init_DataFname")


        graph.add_edge("init_DataFname", "parse_DataFname")
        graph.add_conditional_edges(
            "parse_DataFname",
            self.make_decision_fun,
            {1: "init_chk_DataFname", 2:"init_DataFname"}
        )
        graph.add_edge("init_chk_DataFname", "chk_DataFname")

        graph.add_conditional_edges(
            "chk_DataFname",
            self.make_decision_fun,
            {1: "init_DataID", 2:"init_DataFname",3:"chk_DataFname"}
        )


        graph.add_edge("init_DataID", "parse_DataID")
        graph.add_conditional_edges(
            "parse_DataID",
            self.make_decision_fun,
            {1: "init_chk_DataID", 2:"init_DataID"}
        )
        graph.add_edge("init_chk_DataID", "chk_DataID")

        graph.add_conditional_edges(
            "chk_DataID",
            self.make_decision_fun,
            {1: "init_DataDS", 2:"init_DataID",3:"chk_DataID"}
        )

        graph.add_edge("init_DataDS", "parse_DataDS")
        graph.add_conditional_edges(
            "parse_DataDS",
            self.make_decision_fun,
            {1: "init_chk_DataDS", 2:"init_DataDS"}
        )
        graph.add_edge("init_chk_DataDS", "chk_DataDS")

        graph.add_conditional_edges(
            "chk_DataDS",
            self.make_decision_fun,
            {1: "init_DataREADME", 2:"init_DataDS",3:"chk_DataDS"}
        )

        graph.add_edge("init_DataREADME", "parse_DataREADME")
        graph.add_conditional_edges(
            "parse_DataREADME",
            self.make_decision_fun,
            {1: "init_chk_DataREADME", 2:"init_DataREADME"}
        )
        graph.add_edge("init_chk_DataREADME", "chk_DataREADME")

        graph.add_conditional_edges(
            "chk_DataREADME",
            self.make_decision_fun,
            {1: "init_OS_Time", 2:"init_DataREADME",3:"chk_DataREADME"}
        )
        

        
        graph.add_edge("init_OS_Time", "parse_OS_Time")
        graph.add_conditional_edges(
            "parse_OS_Time",
            self.make_decision_fun,
            {1: "init_chk_OS_Time", 2:"init_OS_Time"}
        )
        graph.add_edge("init_chk_OS_Time", "chk_OS_Time")

        graph.add_conditional_edges(
            "chk_OS_Time",
            self.make_decision_fun,
            {1: "init_OS_Events", 2:"init_OS_Time",3:"chk_OS_Time"}
        )

        

        graph.add_edge("init_OS_Events", "parse_OS_Events")
        graph.add_conditional_edges(
            "parse_OS_Events",
            self.make_decision_fun,
            {1: "init_chk_OS_Events", 2:"init_OS_Events"}
        )
        graph.add_edge("init_chk_OS_Events", "chk_OS_Events")

        graph.add_conditional_edges(
            "chk_OS_Events",
            self.make_decision_fun,
            {1: "init_PFS_Time", 2:"init_OS_Events",3:"chk_OS_Events"}
        )


        
        graph.add_edge("init_PFS_Time", "parse_PFS_Time")
        graph.add_conditional_edges(
            "parse_PFS_Time",
            self.make_decision_fun,
            {1: "init_chk_PFS_Time", 2:"init_PFS_Time"}
        )
        graph.add_edge("init_chk_PFS_Time", "chk_PFS_Time")

        graph.add_conditional_edges(
            "chk_PFS_Time",
            self.make_decision_fun,
            {1: "init_PFS_Events", 2:"init_PFS_Time",3:"chk_PFS_Time"}
        )


        graph.add_edge("init_PFS_Events", "parse_PFS_Events")
        graph.add_conditional_edges(
            "parse_PFS_Events",
            self.make_decision_fun,
            {1: "init_chk_PFS_Events", 2:"init_PFS_Events"}
        )
        graph.add_edge("init_chk_PFS_Events", "chk_PFS_Events")

        graph.add_conditional_edges(
            "chk_PFS_Events",
            self.make_decision_fun,
            {1: "create_newdata", 2:"init_PFS_Events",3:"chk_PFS_Events"}
        )
        
        graph.add_edge("create_newdata", "initQ")
        

        graph.add_edge("init_delete", "parse_delete")
        graph.add_conditional_edges(
            "parse_delete",
            self.make_decision_fun,
            {1: "init_chk_delete", 2:"init_delete"}
        )
        graph.add_edge("init_chk_delete", "chk_delete")

        graph.add_conditional_edges(
            "chk_delete",
            self.make_decision_fun,
            {1: "delete_data", 2:"init_delete",3:"chk_delete"}
        )


        graph.add_edge("delete_data", "initQ")

        self.graph = graph.compile(
            checkpointer=local_memory,
            interrupt_before=["init_Qtypechk_1","parse_DataFname" , "chk_DataFname", "parse_DataID" , "chk_DataID",  "parse_DataDS" , "chk_DataDS", "parse_DataREADME" , "chk_DataREADME" , "parse_OS_Time" , "chk_OS_Time","parse_OS_Events" , "chk_OS_Events" , "parse_PFS_Time" , "chk_PFS_Time","parse_PFS_Events" , "chk_PFS_Events" ,"parse_delete" , "chk_delete"   ]
        )
        
        self.model = model
        self.conversation_buffer =[]
        
        self.DataFname =""
        self.DataID = ""
        self.DataDS = ""
        self.DataREADME = ""
        self.OS_Time = ""
        self.OS_Events = ""
        self.PFS_Time = ""
        self.PFS_Events = ""

        self.delete_ID = ""

        self.html_fname = "dialogs/dm.html"


    
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



   
    def __chk_yn_fun(self, state: AgentState):
        print("__chk_yn_fun")
        messages = state['messages'][-1].content

        with open("dialogs/_yn.pkl", "rb") as f:
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


    def initQ_fun(self, state: AgentState):

        with open('dialogs/_dm_init.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)
        

    def init_Qtypechk_1_fun(self, state: AgentState):
       
        messages = state['messages'][-1].content
        
        with open("dialogs/_dm_initQ.pkl", "rb") as f:
            loaded_chat_prompt = pickle.load(f)

        chain = loaded_chat_prompt | self.model
     
        input_dict = {
            "user_input":messages
    
        }
        
        output = chain.invoke(
            input_dict
        )

        print(output)

    
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
            

        return {'messages': [output]}

    
    def init_create_fun(self, state: AgentState):

        with open('dialogs/_dm_init_create.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        str ="=======================================================\n"+ loaded_dict["title"] +"\n=======================================================\n"
        self.tk_print(str)
        str = loaded_dict["message"]
        self.tk_print(str)

    def init_DataFname_fun(self, state: AgentState):
        self.tk_print("[AI] Please enter the full absolute path and filename for the new dataset, including the .tsv extension (e.g., /path/to/your/folder/clinical_data.tsv).")
    
    def parse_DataFname_fun(self, state: AgentState):
        messages = state['messages'][-1].content
       
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
    
        file_path = messages.strip()
        self.DataFname  = file_path
        try:
            # Try to read the file
            df = pd.read_csv(os.path.normpath(file_path), sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI] Failed to read the file: {e}")
            return {'messages': ["2"]}

        # Check if the file is empty
        if df.empty:
            self.tk_print("[AI] The file is empty.")
            return {'messages': ["2"]}
    

        # Check if there are at least some clinical attribute columns
        if df.shape[1] < 2:
            self.tk_print("The file must have at least one clinical attribute column besides the sample ID or index.")
            return {'messages': ["2"]}
        

        self.tk_print("Valid clinical TSV file!")
        return {'messages': ["1"]}

    def init_chk_DataFname_fun(self, state: AgentState):
        self.tk_print(f"The filename for the new dataset is {self.DataFname}.")   
        self.tk_print("Is that correct? (Yes or No.)") 

    
    def init_DataID_fun(self, state: AgentState):
        self.tk_print("[AI] What is the ID for the new dataset? (less than 10 charactoers.)")
    
    def parse_DataID_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
        words = messages.split()
    
        # Check if there's exactly one word and its length <= 10
        if len(words) == 1 and len(words[0]) < 10:
            self.DataID = words[0]
            output ="1"
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        return {'messages': [output]}

    def init_chk_DataID_fun(self, state: AgentState):
        self.tk_print(f"The ID for the new dataset is {self.DataID}")   
        self.tk_print("Is that correct? (Yes or No.)")   

    def init_DataDS_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide a short description (less than 40 characters) to help you remember what this dataset ID represents.")
    
    def parse_DataDS_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
       
        # Check if there's exactly one word and its length <= 10
        if len(messages) < 40:
            self.DataDS = messages
            output ="1"
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        return {'messages': [output]}

    def init_chk_DataDS_fun(self, state: AgentState):
        self.tk_print(f"The short description for the new dataset is {self.DataDS}")   
        self.tk_print("Is that correct? (Yes or No.)")   

    def init_DataREADME_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide a full description to help users understand what this dataset is about.")
    
    def parse_DataREADME_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
       
        # Check if there's exactly one word and its length <= 10
        if len(messages) < 10000:
            self.DataREADME = messages.strip()
            output ="1"
        else:
            self.tk_print("\n[AI]***WARNING*** Your input is invalid. Please try again.\n")
        return {'messages': [output]}

    def init_chk_DataREADME_fun(self, state: AgentState):
        self.tk_print(f"The full-length description for the new dataset is {self.DataREADME}")   
        self.tk_print("Is that correct? (Yes or No.)")   

    
    def init_OS_Time_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide the column name for overall survival time.")
    
    def parse_OS_Time_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        column_name = messages.strip()
        try:
            # Try to read the file
            df = pd.read_csv(self.DataFname , sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        if column_name not in df.columns:
            self.tk_print(f"[AI]***WARNING*** Column '{column_name}' does not exist in the DataFrame.")
            return {'messages': ["2"]}
    
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column_name]):
            self.tk_print(f"[AI] Column '{column_name}' exists and is numeric.")
            self.OS_Time = column_name
            output ="1"
          
        else:
            self.tk_print(f"\n[AI]***WARNING*** Column '{column_name}' exists but is not numeric.")
            return {'messages': ["2"]}


        return {'messages': [output]}

    def init_chk_OS_Time_fun(self, state: AgentState):
        self.tk_print(f"The column of the overall survival time for the new dataset is {self.OS_Time}")   
        self.tk_print("Is that correct? (Yes or No.)")  

    def init_OS_Events_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide the column name for overall survival events (binary; 0 or 1).")
    
    def parse_OS_Events_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        column_name = messages.strip()
        try:
            # Try to read the file
            df = pd.read_csv(self.DataFname , sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        if column_name not in df.columns:
            self.tk_print(f"[AI]***WARNING*** Column '{column_name}' does not exist in the DataFrame.")
            return {'messages': ["2"]}
    
        # Check if the column is numeric
        values = df[column_name].dropna().unique()

        if  set(values).issubset({0, 1}) :
            self.tk_print(f"[AI] Column '{column_name}' exists and is numeric.")
            self.OS_Events = column_name
            output ="1"
          
        else:
            self.tk_print(f"\n[AI]***WARNING*** Column '{column_name}' exists but is not numeric.")
            return {'messages': ["2"]}


        return {'messages': [output]}

    def init_chk_OS_Events_fun(self, state: AgentState):
        self.tk_print(f"Thecolumn of the overall survival events for the new dataset is {self.OS_Events}")   
        self.tk_print("Is that correct? (Yes or No.)")  

    def init_PFS_Time_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide the column name for progression-free survival time.")
    
    def parse_PFS_Time_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        column_name = messages.strip()
        try:
            # Try to read the file
            df = pd.read_csv(self.DataFname , sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        if column_name not in df.columns:
            self.tk_print(f"[AI]***WARNING*** Column '{column_name}' does not exist in the DataFrame.")
            return {'messages': ["2"]}
    
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column_name]):
            self.tk_print(f"[AI] Column '{column_name}' exists and is numeric.")
            self.PFS_Time = column_name
            output ="1"
          
        else:
            self.tk_print(f"\n[AI]***WARNING*** Column '{column_name}' exists but is not numeric.")
            return {'messages': ["2"]}


        return {'messages': [output]}

    def init_chk_PFS_Time_fun(self, state: AgentState):
        self.tk_print(f"The column of the overall survival time for the new dataset is {self.PFS_Time}")   
        self.tk_print("Is that correct? (Yes or No.)")  

    def init_PFS_Events_fun(self, state: AgentState):
        self.tk_print("[AI] Please provide the column name for progression-free survival events (binary; 0 or 1).")
    
    def parse_PFS_Events_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        output ="2"
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        column_name = messages.strip()
        try:
            # Try to read the file
            df = pd.read_csv(self.DataFname , sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        if column_name not in df.columns:
            self.tk_print(f"[AI]***WARNING*** Column '{column_name}' does not exist in the DataFrame.")
            return {'messages': ["2"]}
    
        # Check if the column is numeric
        values = df[column_name].dropna().unique()

        if  set(values).issubset({0, 1}) :
            self.tk_print(f"[AI] Column '{column_name}' exists and is numeric.")
            self.PFS_Events = column_name
            output ="1"
          
        else:
            self.tk_print(f"\n[AI]***WARNING*** Column '{column_name}' exists but is not numeric.")
            return {'messages': ["2"]}


        return {'messages': [output]}

    def init_chk_PFS_Events_fun(self, state: AgentState):
        self.tk_print(f"Thecolumn of the progression-free survival events for the new dataset is {self.PFS_Events}")   
        self.tk_print("Is that correct? (Yes or No.)")

        
    def create_newdata_fun(self, state: AgentState):
        
       
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        try:
            # Try to read the file
            
            df = pd.read_csv("data/dataset.tsv", sep='\t', header=0 , na_values=missing_values)
            new_row = pd.DataFrame({'Name': [self.DataID], 'Description': [self.DataDS]})

            # Append the new row using concat
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv('data/dataset.tsv', sep='\t', index=False)
            self.tk_print("[AI] update the index file as follows.")   
            self.tk_print(df.to_string())
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            
        
        

        try:
            os.mkdir("data/"+self.DataID)

        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to create the data folder: {e}")

        try:
            output_file = "data/"+self.DataID+"/README.txt" 
            with open(output_file , 'w') as f:
                f.write(self.DataREADME)
            self.tk_print(f"String successfully written to {output_file}.")
        except Exception as e:
            self.tk_print(f"Failed to write to file: {e}")
        try:
            data_dict = {
                "README": "README.txt",
                "DATAFNAME": "pt_metadata.tsv",
                "OS_TIME": self.OS_Time ,
                "OS_STATUS": self.OS_Events, 
                "PFS_TIME": self.PFS_Time ,
                "PFS_STATUS": self.PFS_Events
            }
            output_file = "data/"+self.DataID+"/INDEX.tsv"
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(list(data_dict.items()), columns=["key", "value"])
        
            # Write the DataFrame to a TSV file
            df.to_csv(output_file, sep='\t', index=False)
        
            self.tk_print(f"The INDEX Dictionary successfully written to {output_file}.")
        except Exception as e:
            self.tk_print(f"Failed to write INDEX dictionary to TSV file: {e}")

        df = pd.read_csv(self.DataFname, sep='\t',index_col=0,  header=0 , na_values=missing_values)

        
        pattern = re.compile(r'[^a-zA-Z0-9.]')  # Keep only letters, numbers, and dot (.)

        for col in df.columns:
            try:
                if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                    contains_pipe = df[col].astype(str).str.contains(r'\|').any()

                    if contains_pipe:
                    # Go row by row
                        new_col = []
                        for val in df[col]:
                            if pd.isna(val):
                                new_col.append(val)
                            else:
                                parts = str(val).split('|')
                                cleaned_parts = []
                                for part in parts:
                                    part = part.strip()
                                    cleaned = part.replace(' ', '_')
                                    cleaned_parts.append(cleaned)
                                new_val = '|'.join(cleaned_parts)
                                new_col.append(new_val)
                        df[col] = new_col
                    else:
                        new_col = []
                        for val in df[col]:
                            if pd.isna(val):
                                new_col.append(val)
                            else:
                                cleaned = re.sub(pattern, '_', str(val))
                                new_col.append(cleaned)
                        df[col] = new_col
            except Exception as e:
                self.tk_print(f"Skipping column {col} due to error: {e}")


        output_file = "data/"+self.DataID+"/pt_metadata.tsv"
        df.to_csv(output_file, sep='\t', index=True, na_rep="none")
        


    def init_delete_fun(self, state: AgentState):
        
        
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
        str="\n[AI] What is the name of the dataset you want to delete for the database? Please input the name of the dataset.\n"
        self.tk_print(str)

    
    def parse_delete_fun(self, state: AgentState):
        messages = state['messages'][-1].content
        
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
        row_name = messages.strip()
        try:
            # Try to read the file
            df = pd.read_csv(os.path.normpath("data/dataset.tsv") , sep='\t',index_col=0,  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        if row_name not in df.index:
            self.tk_print(f"[AI]***WARNING*** Column '{row_name}' does not exist in the DataFrame.")
            return {'messages': ["2"]}

        self.delete_ID = row_name

        return {'messages': ["1"]}

    def init_chk_delete_fun(self, state: AgentState):
        self.tk_print(f"[AI] You want to delete {self.delete_ID}")   
        self.tk_print("Is that correct? (Yes or No.)")
    def delete_data_fun(self, state: AgentState):
        
        data_fname = os.path.normpath("data/dataset.tsv") 
        folder_path = os.path.normpath("data/"+self.delete_ID)
        missing_values = ["", "None", "none", "NA", "na", "N/A", "n/a", "NULL", "null", "."]
    
        try:
            # Try to read the file
            df = pd.read_csv(data_fname, sep='\t',  header=0 , na_values=missing_values)
        except Exception as e:
            self.tk_print(f"[AI]***WARNING***  Failed to read the file: {e}")
            return {'messages': ["2"]}

        try:
            row_name = self.delete_ID
            first_col = df.columns[0]  # get the name of the first column
            # Find rows where the first column equals the given row_name
            matching_rows = df[df[first_col] == row_name]    
            if not matching_rows.empty:
            # Drop those rows
                df = df.drop(matching_rows.index)
                df.to_csv(data_fname, sep='\t', index=False, na_rep="none")
                subprocess.run(['rm', '-rf', folder_path], check=True)
            
        except Exception as e:
            print(f"Error removing row: {e}")


    def pop_messages(self):
        if(len(self.conversation_buffer) ==0):
            return ""

        output = "\n".join(self.conversation_buffer)
        self.conversation_buffer=[] 
        return output

    def start(self,thread, thread_id , conversation_path):
         
        # self.conversation_path = conversation_path 
        # if not os.path.exists(self.conversation_path):
        #     os.makedirs(self.conversation_path)
        
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
        
        if snapshot.next[0]=="parse_DataFname" :
            node_str =  "init_DataFname"
        if snapshot.next[0]=="chk_DataFname" :
            node_str =  "init_chk_DataFname"

        if snapshot.next[0]=="parse_DataID" :
            node_str =  "init_DataID"
        if snapshot.next[0]=="chk_DataID" :
            node_str =  "init_chk_DataID"

        if snapshot.next[0]=="parse_DataDS" :
            node_str =  "init_DataDS"
        if snapshot.next[0]=="chk_DataDS" :
            node_str =  "init_chk_DataDS"

        if snapshot.next[0]=="parse_DataREADME" :
            node_str =  "init_DataREADME"
        if snapshot.next[0]=="chk_DataREADME" :
            node_str =  "init_chk_DataREADME"
        
        
        if snapshot.next[0]=="parse_OS_Time" :
            node_str =  "init_OS_Time"
        if snapshot.next[0]=="chk_OS_Time" :
            node_str =  "init_chk_OS_Time"
        
        if snapshot.next[0]=="parse_OS_Events" :
            node_str =  "init_OS_Events"
        if snapshot.next[0]=="chk_OS_Events" :
            node_str =  "init_chk_OS_Events"

        if snapshot.next[0]=="parse_PFS_Time" :
            node_str =  "init_PFS_Time"
        if snapshot.next[0]=="chk_PFS_Time" :
            node_str =  "init_chk_PFS_Time"
        
        if snapshot.next[0]=="parse_PFS_Events" :
            node_str =  "init_PFS_Events"
        if snapshot.next[0]=="chk_PFS_Events" :
            node_str =  "init_chk_PFS_Events"
        
        if snapshot.next[0]=="parse_delete" :
            node_str =  "init_delete"
        if snapshot.next[0]=="chk_delete" :
            node_str =  "init_chk_delete"
        
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

    abot = Data_manager( llm, memory_p1  )
    
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
    conversation_content =abot.pop_messages()
    print(conversation_content)
        
    print("Goodbye!")
       
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
    from PIL import Image
    from io import BytesIO

    
    image_stream = BytesIO(abot.graph.get_graph().draw_png())

    # Open the image using PIL
    image = Image.open(image_stream)

    image.save("saved_image.png")

if __name__=="__main__":
    

    main()
