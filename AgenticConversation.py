
import requests
import json
import autogen
import smtplib
import chromadb
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

from autogen import Agent
import streamlit as st
from customStyle import chat_styles 


from typing_extensions import Annotated

# Inject CSS from customStyles.py
st.markdown(chat_styles(), unsafe_allow_html=True)

# Title
st.title("💬 Agentic AI")

llm_config = { "config_list": [{ "model": "gpt-4o-mini",    "api_type": "azure", "api_key": '' ,"base_url":"https://agenttsgpoc.openai.azure.com/",    "api_version": "2024-12-01-preview"}] }

col1, col2 = st.columns(2)

# Left-side chat input 
with col1:
    st.subheader("👤 TSG-Agent")
    tsg_input = st.text_area("Type your TSG steps for AssistantAgent:", key="left_input")

# Right-side chat input
with col2:
    st.subheader("🤖 ICM-Agent")
    incident_input = st.text_area("Type your input to perform:", key="right_input")

tsg_assistant_system_message = "You are a troubleshooting guid (TSG)."
incident_assistant_system_message = """You are a incident manager.
Explain each step and send kusto query to tool agent for execution. Send one query at a time.
Reply TERMINATE when done or error out.  
Reflect on your answer, and if you think you are hallucinating, or repeating then reply `TERMINATE`"""

user_proxy_system_message = "You are code executor tool"

PROBLEM = incident_input

tsg_assistant = AssistantAgent(
    name = "tsg_assistant",
    human_input_mode="NEVER", 
    llm_config=llm_config,
    max_consecutive_auto_reply=15,
    system_message = tsg_assistant_system_message
    )

incident_assistant = AssistantAgent(
    name="incident_assistant",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    system_message= incident_assistant_system_message,
    description = "I am incident manager. I will summarize the steps and provide instruct to execute"
)

user_proxy = UserProxyAgent(
    name = "user_proxy",
    human_input_mode="NEVER",
    llm_config=False,
    code_execution_config=False,
    max_consecutive_auto_reply=15,
    system_message = user_proxy_system_message,
    #is_termination_msg=lambda msg: "Thank you or terminate" in msg["content"],
    default_auto_reply="Reply `TERMINATE` if the task is completed or if there is any error.",
    description = "I am a code executor tool. I will only execute kusto query"
    )

rag_user_proxy = RetrieveUserProxyAgent(
    name="rag_user_proxy",
    system_message=user_proxy_system_message,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": tsg_input,
        #"extra_docs": True,
        "update_context" : False,
        "chunk_token_size": 3000,
        "model": "gpt-4o-mini",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": chromadb.PersistentClient(path="/tmp/chromadb"),
        #"client": client,
        "collection_name": "groupchat",
        "get_or_create": True,
        #"overwrite": True,
        #"distance_threshold": 0
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

# Execute kusto query
def run_kusto_query(query: str):    
    url = "http://localhost:9000/v1/rest/query"
    headers = {
        "Content-Type": "application/json"
    }    
    body = {
        "db": "NetDefaultDB",
        "csl": query
    }
    json_body = json.dumps(body)
    response = requests.post(url, headers=headers, data=json_body)
    return response.text

# Send Email
def run_email_query(body: str):    
    smtp_server = "localhost"
    smtp_port = 1025  # Same as in the debugging server

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.sendmail(
        "from@example.com", 
        "to@example.com", 
        "Subject: Test Email\n\nHello, this is a test!  "+body)
    server.quit()
    
    return "Email sent successfully !!"


# Register Custom Function for Query Execution
@user_proxy.register_for_execution()
@incident_assistant.register_for_llm(description="retrieve kusto query and execute one query at a time")
def execute_kusto(query: Annotated[str, "kusto query"]):
    try:
        return run_kusto_query(query)
    except Exception as e:
        return str(e)

# Register Custom Function for Query Execution
#@user_proxy.register_for_execution()
#@tsg_assistant.register_for_llm(description="Send summary in email ")
def execute_email(email_content: Annotated[str, "email body"]):
    try:
        return run_email_query(email_content)
    except Exception as e:
        return str(e)

def _reset_agents():
    rag_user_proxy.reset()
    tsg_assistant.reset()
    incident_assistant.reset()    
    user_proxy.reset()

def retrieve_content(
        message: Annotated[
            str,            
            "Refined message which keeps the original meaning and can be used to retrieve content for code and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        rag_user_proxy.n_results = n_results  # Set the number of results to be retrieved.
        #_context = {"problem": PROBLEM, "n_results": n_results, "search_string": ""}
        _context = {"problem": message, "n_results": n_results}
        ret_msg = rag_user_proxy.message_generator(rag_user_proxy, None, _context)
        return ret_msg or message

rag_user_proxy.human_input_mode = "NEVER"  

for caller in [tsg_assistant]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content and question answering.", api_style="tool"
        )(retrieve_content)

for executor in [user_proxy]:
        executor.register_for_execution()(d_retrieve_content)


counter = 0

def custom_speaker_selection_func(last_speaker: Agent, groupchat: autogen.GroupChat):
    global counter

    messages = groupchat.messages

    if last_speaker is user_proxy and counter == 0:
        counter = 1        
        return tsg_assistant
    elif last_speaker is tsg_assistant :
        return user_proxy
    elif last_speaker is incident_assistant:
        return user_proxy
    elif last_speaker is user_proxy:
        return incident_assistant
    else:
        return "random"

groupchat = autogen.GroupChat(
        agents=[tsg_assistant,incident_assistant,user_proxy],
        messages=[],
        max_round=30,
        speaker_selection_method=custom_speaker_selection_func,
        allow_repeat_speaker=False,
    )

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# Button to submit input
if st.button("Submit"):
    _reset_agents()

    chat_result=user_proxy.initiate_chat(
        manager,
        message=PROBLEM,
    )

    data = chat_result.chat_history
  
    # Extract required fields
    fields_to_extract = ["role", "name", "tool_calls"]

    parsed_output = []
    for item in data:
        parsed_item = {field: item.get(field) for field in fields_to_extract}
        content_value = item.get("content", "")

        # Check if content has TableName = Table_0
        if isinstance(content_value, str) and "Table_0" in content_value:
            try:
                json_data = json.loads(content_value)
                for table in json_data.get("Tables", []):
                    if table.get("TableName") == "Table_0":
                        table_0_data = table
                        break

                if table_0_data:
                    parsed_item["content"] = table_0_data
                    #print(json.dumps(table_0_data, indent=2))
                else:
                    parsed_item["content"] = "Table_0 not found in content."
                    #print("Table_0 not found in content.")
            except Exception as e:
                parsed_item["content"] = f"[Error extracting Table_0: {str(e)}]"
        else:
            parsed_item["content"] = content_value

        parsed_output.append(parsed_item)

    with st.container():
        
        for i, item in enumerate(parsed_output, 1):
            
            output = ""
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    formatted_value = json.dumps(value, indent=2)
                else:
                    formatted_value = str(value)

                output += f"<b>{key}:</b>{formatted_value}<br>"

            if i%2 != 0 :
                st.markdown(f'<div class="user-message">  {output} </div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{output}</div>', unsafe_allow_html=True)


