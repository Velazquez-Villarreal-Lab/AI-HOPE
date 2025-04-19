# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama.llms import OllamaLLM
from Data_manager import Data_manager

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

class AI_DM:

    def __init__(self, root):
        self.root = root
        self.user_input = tk.StringVar()
        self.html_fname = "dialogs/dm.html"
        
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
   
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                html_content = file.read()
            self.html_viewer.set_html(html_content)
        else:
            self.output_text.insert(tk.END, f"File not found: {file_path}")

    def on_submit(self):
        
        input_str = self.input_text.get("1.0", tk.END).strip() 

        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, f"[AI] Your input is : {input_str}\n")
        self.output_text.insert(tk.END, "[AI] Processing your input ...\n\n")
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
    
    def run(self):
        user_input = ""
        current_directory = os.getcwd()
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(current_directory+"/conversations/"+current_time)

        self.conversation_path = current_directory+"/conversations/"+current_time
        user_input = ""
        ### init llm agent
        # Define the LLM
        llm =  OllamaLLM(model="llama3",temperature=0)
        thread_p1 = {"configurable": {"thread_id": "1"}}
        memory_p1 = MemorySaver()

        abot = Data_manager( llm, memory_p1  )
        abot.start(thread_p1, "1" , self.conversation_path)
        
        while True :
            conversation_content =abot.pop_messages()
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, conversation_content+"\n")
            self.output_text.config(state=tk.DISABLED)
            self.output_text.see(tk.END)
            
            self.display_html(self.html_fname )
            
            self.user_input.set('')   
            self.root.wait_variable(self.user_input)

            input_str = self.user_input.get()
            
            if input_str.lower() in ["quit", "exit", "q"]:
                break
            if ( not abot.proceed(thread_p1, "1",input_str) ) :
                break
        self.root.quit() 
        self.root.destroy() 
        print("Goodbye!")
        print("All the statistical reports are generated at " + self.conversation_path  + ".")
     


root = tk.Tk()
root.title("AI Agent for Clinical Data Management")

# Make the window resizable
root.geometry("1280x960")
root.minsize(300, 200)

# Create an instance of the Agent class

gui = AI_DM(root)

# Start the keep_asking method in the Tkinter event loop
root.after(1000, gui.run())

# Start the Tkinter event loop
root.mainloop()