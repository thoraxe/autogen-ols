import asyncio
import logging
import random
import subprocess
import os
import sys

import colorlog
from devtools import pprint

from typing import AsyncIterator

import autogen_agentchat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    HandoffTermination,
    MaxMessageTermination,
    TextMentionTermination,
)
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
from autogen_ext.models.openai import OpenAIChatCompletionClient

log_format = "%(log_color)s%(asctime)s [%(levelname)s] %(reset)s%(purple)s[%(name)s] %(reset)s%(blue)s%(message)s"
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(log_format))

logging.basicConfig(level=logging.WARNING, handlers=[handler])

ourlogger = logging.getLogger(__name__)
# logger = logging.getLogger(TRACE_LOGGER_NAME)
# logger2 = logging.getLogger(autogen_agentchat.TRACE_LOGGER_NAME)
# logger.setLevel(logging.INFO)
# logger2.setLevel(logging.INFO)
ourlogger.setLevel(logging.INFO)

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

azure_api_key = os.getenv("AZURE_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
azure_model = os.getenv("AZURE_MODEL")
azure_api_version = os.getenv("AZURE_API_VERSION")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")

pre_path = "/home/thoraxe/bin/"


#default_client = OpenAIChatCompletionClient(model=openai_model, api_key=openai_api_key)

default_client = AzureOpenAIChatCompletionClient(
   azure_deployment=azure_deployment,
   model=azure_model,
   api_version=azure_api_version,
   azure_endpoint=azure_endpoint,
   api_key=azure_api_key,
)


#### TOOLS ####
retrieval_agent_tool_list = []


async def get_namespaces() -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", "namespaces"], capture_output=True, timeout=2
    )
    return output.stdout


get_namespaces_function_tool = FunctionTool(
    get_namespaces, description="Fetch the list of all namespaces in the cluster"
)

retrieval_agent_tool_list.append(get_namespaces_function_tool)

# async def get_pod_list(namespace: str) -> str:
#    output = subprocess.run(
#        [pre_path + "oc", "get", "pods", "-n", namespace, "-o", "name"],
#        capture_output=True,
#        timeout=2,
#    )
#    return output.stdout
#
#
# get_pod_list_function_tool = FunctionTool(
#    get_pod_list, description="Fetch a list of the pods in a specific namespace"
# )
# retrieval_agent_tool_list.append(get_pod_list_function_tool)


# note - would fail if user is not cluster admin
async def get_object_cluster_wide_list(kind: str) -> str:
    ourlogger.info(f"provided kind: {kind}")
    output = subprocess.run(
        [pre_path + "oc", "get", kind, "-A", "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_object_cluster_wide_list_function_tool = FunctionTool(
    get_object_cluster_wide_list,
    description="""
    Fetch a list of all instances of a specific type of kubernetes/openshift 
    object in the cluster.

    Args:
        kind: the kubernetes/openshift objects to get
    """,
)
retrieval_agent_tool_list.append(get_object_cluster_wide_list_function_tool)


async def get_object_namespace_list(kind: str, namespace: str) -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", kind, "-n", namespace, "-o", "name"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_object_namespace_list_function_tool = FunctionTool(
    get_object_namespace_list,
    description="""
    Fetch a list of all instance of a specific type of kubernetes/openshift
    object in a specific namespace.

    Args:
        kind: the kubernetes/openshift objects to get
        namespace: the namespace containing the objects
    """,
)
retrieval_agent_tool_list.append(get_object_namespace_list_function_tool)


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
retrieval_agent_tool_list.append(get_nonrunning_pods_function_tool)


async def get_object_details(namespace: str, kind: str, name: str) -> str:
    output = subprocess.run(
        [pre_path + "oc", "get", kind, "-n", namespace, name, "-o", "yaml"],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_object_details_function_tool = FunctionTool(
    get_object_details,
    description="""
    Fetch the details for a specific object in the cluster.

    Args:
        namespace: the namespace where the object is
        kind: the kind of the object
        name: the name of the object
    """,
)
retrieval_agent_tool_list.append(get_object_details_function_tool)


async def get_pod_status(namespace: str, pod: str) -> str:
    output = subprocess.run(
        [
            pre_path + "oc",
            "get",
            "pod",
            "-n",
            namespace,
            pod,
            "-o",
            "jsonpath='\{.status\}'",
        ],
        capture_output=True,
        timeout=2,
    )
    return output.stdout


get_pod_status_function_tool = FunctionTool(
    get_pod_status,
    description="""
    Fetch the status information for a specific pod in the cluster by namespace.
    Only returns pod status and no other details.

    Args:
        namespace: the namespace where the pod is
        pod: the name of the pod to get the status for
    """,
)
retrieval_agent_tool_list.append(get_pod_status_function_tool)

def get_object_health(kind: str, name: str, namespace: str) -> str:

    ourlogger.info(f"get_object_health: ns:{namespace} {kind}/{name}")
    ourlogger.info(f"namespace type is {type(namespace)}")

    if type(namespace) is str:
        if namespace == "":
            output = subprocess.run(f"{pre_path}kube-health -H {kind}/{name}",
                                    shell=True,
                                    capture_output=True,
                                    timeout=2
                                    )
        else:
            output = subprocess.run(f"{pre_path}kube-health -n {namespace} -H {kind}/{name}",
                                    shell=True,
                                    capture_output=True,
                                    timeout=2
                                    )
    
    nlines = len(output.stdout.splitlines())

    if nlines < 2:
        return "Error: The object you are looking for does not exist"
    else:
        return output.stdout

get_object_health_function_tool = FunctionTool(
    get_object_health,
    description="""
    A simple tool to describe the health of an object in the cluster. Must be
    used on individual objects one at a time. Does not accept 'all' as a name.
    For example, if you want to look at all nodes, you must run this tool one at
    a time against each individual node.

    Args:
      namespace (str): the namespace where the object is. for a cluster-scoped object, use None
      kind (str): the type of object
      name (str): the name of the object

    Returns:
      str: text describing the health of the object
    """,
)
retrieval_agent_tool_list.append(get_object_health_function_tool)

async def retrieval_tool(query: str) -> None:
    await Console(
        retrieval_agent.on_messages_stream(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
    )


async def knowledge_tool(query: str) -> None:
    await Console(
        knowledge_agent.on_messages_stream(
            [TextMessage(content=query, source="user")],
            cancellation_token=CancellationToken(),
        )
    )

#### END TOOLS ####

#### AGENTS ####
routing_agent = AssistantAgent(
    name="routing_agent",
    handoffs=["retrieval_agent", "knowledge_agent"],
    system_message="""You are an agent-picking agent whose only job is to choose
which agent to pass questions to.

The retrieval_agent is in charge of getting information from an openshift or
kubernetes cluster.
The knowledge_agent is in charge of answering
knowledge-based questions like how-to, documentation, and related questions.

Do not summarize responses from your agents. Be sure to pass their output directly
to the user.

After all tasks are complete, summarize the findings and end with "TERMINATE".""",
    model_client=default_client,
)

retrieval_agent = AssistantAgent(
    name="retrieval_agent",
    handoffs=["routing_agent"],
    description="This agent is used for retrieving data from OpenShift and Kubernetes clusters.",
    system_message="""You are a Kubernetes and OpenShift assistant. You should
only answer questions related to OpenShift and Kubernetes. You can retrieve
information from Kubernetes and OpenShift environments using your tools.

When the transaction is complete, handoff to the routing_agent to finalize.

In general:
* when it can provide extra information, first run as many tools as you need to gather more information, then respond. 
* if possible, do so repeatedly with different tool calls each time to gather more information.
* do not stop investigating until you are at the final root cause you are able to find. 
* use the "five whys" methodology to find the root cause.
* for example, if you found a problem in microservice A that is due to an error in microservice B, look at microservice B too and find the error in that.
* if you cannot find the resource/application that the user referred to, assume they made a typo or included/excluded characters like - and.
* in this case, try to find substrings or search for the correct spellings
* if you are unable to investigate something properly because you do not have access to the right data, explicitly tell the user that you are missing an integration to access XYZ which you would need to investigate. you should specifically use the templated phrase "I don't have access to <details>."
* always provide detailed information like exact resource names, versions, labels, etc
* even if you found the root cause, keep investigating to find other possible root causes and to gather data for the answer like exact names
* if a runbook url is present as well as tool that can fetch it, you MUST fetch the runbook before beginning your investigation.
* if you don't know, say that the analysis was inconclusive.
* if there are multiple possible causes list them in a numbered list.
* there will often be errors in the data that are not relevant or that do not have an impact - ignore them in your conclusion if you were not able to tie them to an actual error.
* run as many kubectl commands as you need to gather more information, then respond.
* if possible, do so repeatedly on different Kubernetes objects.
* for example, for deployments first run kubectl on the deployment then a replicaset inside it, then a pod inside that.
* when investigating a pod that crashed or application errors, always run kubectl_describe and fetch logs with both kubectl_previous_logs and kubectl_logs so that you see current logs and any logs from before a crash.
* do not give an answer like "The pod is pending" as that doesn't state why the pod is pending and how to fix it.
* do not give an answer like "Pod's node affinity/selector doesn't match any available nodes" because that doesn't include data on WHICH label doesn't match
* if investigating an issue on many pods, there is no need to check more than 3 individual pods in the same deployment. pick up to a representative 3 from each deployment if relevant
* if you find errors and warning in a pods logs and you believe they indicate a real issue. consider the pod as not healthy. 
* if the user says something isn't working, ALWAYS:
** use kubectl_describe on the owner workload + individual pods and look for any transient issues they might have been referring to
** check the application aspects with kubectl_logs + kubectl_previous_logs and other relevant tools
** look for misconfigured ingresses/services etc

Style guide:
* Be painfully concise.
* Leave out "the" and filler words when possible.
* Be terse but not at the expense of leaving out important data like the root cause and how to fix.
""",
    model_client=default_client,
    tools=retrieval_agent_tool_list,
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
    model_client=default_client,
)

# Define termination condition
user_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")
max_term = MaxMessageTermination(max_messages=25)
termination = text_termination | max_term | user_termination

team = Swarm(
    [routing_agent, retrieval_agent, knowledge_agent], termination_condition=termination
)

user_query = "Are there any unhealthy pods in my cluster?"
user_query = "What pods are in the namespace openshift-lightspeed?"
user_query = "How do I scale a pod automatically?"


async def assistant_run() -> None:

    print("----START----")
    task_result = await Console(team.run_stream(task=sys.argv[1]))


# Use asyncio.run(assistant_run()) when running in a script.
asyncio.run(assistant_run())
