import asyncio
import gradio as gr
import logging
import random
import subprocess
import os

from typing import AsyncIterator

import autogen_agentchat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from autogen_core import TRACE_LOGGER_NAME
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(TRACE_LOGGER_NAME)
#logger2 = logging.getLogger(autogen_agentchat.TRACE_LOGGER_NAME)
#logger.setLevel(logging.INFO)
#logger2.setLevel(logging.INFO)

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
azure_model = os.getenv("AZURE_MODEL")
azure_api_version = os.getenv("AZURE_API_VERSION")

pre_path = "/home/thoraxe/bin/"

default_client = AzureOpenAIChatCompletionClient(
    azure_deployment=azure_deployment,
    model=azure_model,
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)


#### TOOLS ####
async def get_namespaces() -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", "namespaces"], capture_output=True, timeout=2
    )
    return output.stdout


get_namespaces_function_tool = FunctionTool(
    get_namespaces, description="Fetch the list of all namespaces in the cluster"
)


async def get_pod_list(namespace: str) -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", "pods", "-n", namespace, "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_pod_list_function_tool = FunctionTool(
    get_pod_list, description="Fetch a list of the pods in a specific namespace"
)


async def get_nonrunning_pods() -> str:
    output = subprocess.run(
        [
            pre_path + "oc",
            "get",
            "pods",
            "-A",
            "--field-selector",
            "status.phase!=Running",
            "-o",
            "custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name",
        ],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_nonrunning_pods_function_tool = FunctionTool(
    get_nonrunning_pods,
    description="Fetch a list of pods that are not currently running",
)


async def get_pod_details(namespace: str, pod: str) -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", "pod", "-n", namespace, pod, "-o", "yaml"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_pod_details_function_tool = FunctionTool(
    get_pod_details,
    description="Fetch the details for a single, specific pod in the cluster",
)


async def retrieval_tool(query: str) -> None:
    print("RETRIEVAL_TOOL")
    ## start with a fresh agent
    # await retrieval_agent.on_reset(cancellation_token=CancellationToken())

    await Console(
        retrieval_agent.on_messages_stream(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
    )


async def knowledge_tool(query: str) -> None:
    print("KNOWLEDGE_TOOL")
    ## start with a fresh agent
    # await knowledge_agent.on_reset(cancellation_token=CancellationToken())

    await Console(
        knowledge_agent.on_messages_stream(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
    )


retrieval_agent_tool_list = [
    get_namespaces_function_tool,
    get_pod_list_function_tool,
    get_nonrunning_pods_function_tool,
    get_pod_details_function_tool,
]

#### END TOOLS ####

#### AGENTS ####
routing_agent = AssistantAgent(
    name="routing_agent",
    handoffs=["retrieval_agent","knowledge_agent","user"],
    system_message="""You are an agent-picking agent whose only job is to choose
which agent to pass questions to.

The retrieval_agent is in charge of getting information from an openshift or
kubernetes cluster.
The knowledge_agent is in charge of answering
knowledge-based questions like how-to, documentation, and related questions.

After all tasks are complete, summarize the findings and end with "TERMINATE".""",
    model_client=default_client
)

retrieval_agent = AssistantAgent(
    name="retrieval_agent",
    handoffs=["routing_agent"],
    description="This agent is used for retrieving data from OpenShift and Kubernetes clusters.",
    system_message="""You are a Kubernetes and OpenShift assistant. You should
only answer questions related to OpenShift and Kubernetes. You can retrieve
information from Kubernetes and OpenShift environments using your tools.

When the transaction is complete, handoff to the routing_agent to finalize.
""",
    model_client=default_client,
    tools=retrieval_agent_tool_list
)

knowledge_agent = AssistantAgent(
    name="knowledge_agent",
    handoffs=["routing_agent"],
    description="""An agent used for answering general knowledge, how-to,
documentation, and other similar questions about OpenShift and Kubernetes""",
    system_message="""You are a Kubernetes and OpenShift assistant. You should
only answer questions related to OpenShift and Kubernetes. You are supposed
to answer general knowledge, how-to, documentation, and other similar
questions about OpenShift and Kubernetes

When the transaction is complete, handoff to the routing_agent to finalize.
""",
    model_client=default_client
)

# Define termination condition
user_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")
max_term = MaxMessageTermination(max_messages=25)
termination = text_termination | max_term | user_termination

team = Swarm(
    [routing_agent, retrieval_agent, knowledge_agent],
    termination_condition=termination
)

user_query = "Are there any unhealthy pods in my cluster?"
user_query = "How do I scale a pod automatically?"
user_query = "What pods are in the namespace openshift-lightspeed?"


async def assistant_run() -> None:

    print("----START----")
    task_result = await Console(team.run_stream(task=user_query))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

    # print("\n\n")
    # print("--- ROUTING AGENT ---")
    # print(response.inner_messages)

    # print("\n--- THE FINAL ANSWER ---")
    # print(response.chat_message)


# Use asyncio.run(assistant_run()) when running in a script.
asyncio.run(assistant_run())
