import pickle

# msg_dict = {
#     "title":"Guideline for Case-Control Bioinformatic Data Analysis Using LLM Agent",
#     "message":"This workflow helps users select case and control samples and perform four core bioinformatics analyses: odds ratio test, gene expression analysis, pathway enrichment, and survival analysis. Follow these steps to ensure a smooth and efficient analysis process.\n\n* Select case and control samples, ensuring they are well-defined and relevant to your study.\n* Use bioinformatics agents to run the chosen analysis.\n* The LLM agents will create HTML reports to summarize the analytic results for easy interpretation and sharing.\n\n"
# }

# with open('dialogs/init_case_1.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)

msg_dict = {
    "title":"Set Up Case Samples",
    "message":"To begin, you must provide the name of the dataset for the case samples. Once the name is entered, the agent will proceed with sample selection and analysis setup.\n\n"
}

with open('dialogs/init_case_1.pkl', 'wb') as f:
    pickle.dump(msg_dict, f)




# msg_dict = {
#     "title":"Input Case Dataset Name",
#     "message":"To begin, you must provide the name of the dataset for the case samples. Once the name is entered, the agent will proceed with sample selection and analysis setup.\n\n"
# }

# with open('dialogs/init_case_2.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


msg_dict = {
    "title":"Decide Your Next Action",
    "message":'You have the following options to move forward:\n\n*** Set Up Criteria: You want to filter the dataset based on specific criteria. You can say something like, "I want to filter the cohort" or "Set up filtering criteria."\n\n*** Proceed: If you are ready to move forward without applying any other filters, you can say something like, "I am ready to proceed" or "Let us move forward." If no criteria are set, all samples in the dataset will be included in the analysis.\n\nIf you are not familiar with the dataset, you can explore the data using the following option.\n\n*** Explore Data: If you would like to check the data attributes and value distributions, you can say something like, "Show me the data attributes" or "I want to see the available clinical attributes."\n\n'
    }

with open('dialogs/parse_query_I.pkl', 'wb') as f:
    pickle.dump(msg_dict, f)

# msg_dict = {
#     "title":"A guideline for setting up criteria to refine samples.",
#     "message": 'Now, you can define criteria to refine your samples. Provide clauses that specify relationships between data attributes, comparison operators, and values.\n\n*** Each sentence you provide should represent a valid logical expression. \n\n "Age is greater than 30,"\n\nA valid sentence for defining criteria includes a data attribute, a comparison operator, and a value. For example, in the sentence "Age is greater than 30," "Age" is the data attribute, "is greater than" is the comparison operator, and 30 is the valid value. Common comparison operators include "is," "is not," "greater than," "less than," "greater than or equal to," and "less than or equal to." You can define a range using "from [Start Value] to [End Value]" (e.g., "Age is from 10 to 20"). To specify inclusion or exclusion of a set of values, use "is in" or "is not in" with brackets (e.g., "Disease stage is in {stage I, stage II, stage III}").\n\n*** Each clause should be enclosed in parentheses for clarity.\n\n "(Age is greater than 30) and (Gender is male) or (Diagnosis is cancer)"\n\nWhen there are multiple clauses, each clause should be enclosed in parentheses for clarity. Users can connect multiple logical expressions using "and" or "or." For example, you could specify "(Age is greater than 30) and (Gender is male) or (Diagnosis is cancer) or (Smoking status is not in {current smoker, former smoker})." If needed, nested parentheses can be used to set the priority of evaluation, with inner expressions being evaluated first.\n\nWhat is the criteria you would like to use to refine your samples?'
# }

# with open('dialogs/set_up_criteria.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


# msg_dict = {
#     "title":"Set Up Control Samples",
#     "message":"We are now working on the Control samples. Please provide the name of the dataset for the control samples. Once the name is entered, the agent will proceed with selecting the samples and setting up the analysis.\n\n"
# }

# with open('dialogs/init_ctrl_1.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)

# msg_dict = {
#     "title":"Select an Analysis Option",
#     "message":"You have successfully defined the case and control groups. Now, you can choose the type of analysis you would like to run. Please select one of the following options:\n\n* Odds Ratio Test: Test the odds ratio based on clinical conditions. Example: Perform an odds ratio test.\n\n* Gene Expression: Analyze gene expression in the case and control groups. Example: Run gene expression analysis.\n\n* GSEA (Gene Set Enrichment Analysis): Run GSEA for the selected cohorts. Example: Perform Gene Set Enrichment Analysis.\n\n* Survival Analysis: Conduct survival analysis based on the cohort data. Example: Analyze survival of patients in the two cohorts.\n\n What analysis would you like to select to continue?\n"
# }

# with open('dialogs/init_exec.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


# msg_dict = {
#     "title":"Select an Analysis Option",
#     "message":'You have successfully defined the case and control groups. Now, you can choose the type of analysis you would like to run. Please select one of the following options:\n\n* Odds Ratio Test: Test the odds ratio based on clinical conditions. Example: "Perform an odds ratio test."\n\n* Survival Analysis: Conduct survival analysis based on the cohort data. Example: "Analyze survival of patients in the two cohorts."\n\n What analysis would you like to select to continue?\n'
# }

# with open('dialogs/init_exec.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


# msg_dict = {
#     "title":"Define the Context for Odds Ratio Test",
#     "message":'To perform the Odds Ratio Test, define the context you want to compare between the case and control groups, similar to how you defined the groups. Identify the data attribute (e.g., smoking status, disease presence), specify the criteria using comparison operators (e.g., "Smoking status is in {current smoker, former smoker}" or "Age is greater than 50"), and ensure it applies to both case and control groups. Once the context is set, you can proceed with the test.\n\nWhat context would you like to define for the Odds Ratio Test?'
# }

# with open('dialogs/init_OR.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)

# msg_dict = {
#     "title":"Comprehensive Survival Data Analysis for Case-Control Studies",
#     "message":'In this survival data analysis, we will leverage the case-control stratification you provided to generate Kaplan-Meier (KM) plots for Overall Survival (OS) and Progression-Free Survival (PFS), offering a visual summary of survival outcomes based on your defined stratification. Additionally, we will conduct Cox regression to calculate hazard ratios, assessing the influence of the case-control groups and other key variables on survival outcomes.\n\n[AI] Would you like to proceed with a univariate analysis based solely on the case-control groups? (Yes or No.) If you choose "No," we will proceed with a multivariate analysis, incorporating additional variables to explore how multiple factors influence survival.\n\n'
# }

# with open('dialogs/init_Survival.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


# msg_dict = {
#     "title":"Guideline for Providing Data Attributes for Multivariate Survival Analysis",
#     "message":'To proceed with a multivariate survival analysis, please provide the names of the data attributes you would like to include as additional factors. These attributes will be used alongside the primary case-control groups to analyze how multiple variables influence survival outcomes. List each attribute name you want to include, separated by a comma.\nE.g., age, gender, tumor_stage, treatment_type.\n\n'
# }

# with open('dialogs/multi_Survival.pkl', 'wb') as f:
#     pickle.dump(msg_dict, f)


