import torch
from typing import Sequence
from response_generator.tools.FaasTool import FaasTool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import (
    RunnablePassthrough,
    Runnable
)
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_mistralai.chat_models import ChatMistralAI

class Response_Generator:

    def __init__(self):
        self.load_environment()
        self.print_device_debug_info()

        self.setup_llm()
        self.setup_agent_prompt_template()
        self.setup_memory()
        self.setup_tools()
        self.setup_agent()

    def load_environment(self):
        load_dotenv()

    def print_device_debug_info(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device Info:", device)
        print(torch.__version__)
        print(torch.version.cuda)
        print(torch.randn(1).cuda())

    def setup_llm(self):

        # This is used when loading the model locally (GGUF)
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # n_gpu_layers = 0 # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        # n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(base_dir, "./models/B-Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf")

        # self.llm = LlamaCpp(
        #     model_path=model_path,
        #     temperature=0,
        #     max_tokens=2000,
        #     top_p=1,
        #     callback_manager=callback_manager,
        #     verbose=True,  # Verbose is apparently required for Mixtral to pass to the callback manager
        #     n_ctx=2048
        # )

        # Dont forget to set the API key in .env
        self.llm = ChatMistralAI(model="mistral-small")

    def setup_agent_prompt_template(self):
        template = """
            <s> [INST]
            Respond to the human as helpfully and accurately as possible. You have access to the following tools:

            {tools}
            
            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
            "action": $TOOL_NAME,
            "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
            "action": "Final Answer",
            "action_input": "has to be a single string!"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. You don't have to escape characters. Use tools if necessary. Format is Action:```$JSON_BLOB```then Observation

            {chat_history}

            {input}

            {agent_scratchpad}
            (reminder to respond in a JSON blob no matter what)
            
            [/INST]
        """
        self.agent_prompt_template = ChatPromptTemplate.from_template(template)

    def setup_memory(self):
        # This memory is so that the LLM actually remembers the last couple of messages, for more info see: https://python.langchain.com/docs/modules/memory/
        self.memory = ConversationBufferWindowMemory(k=3, return_messages=False) # return_messages --> Messages as array (True) or single string (False)

    def load_memory(self, current_chain_dict):
        return self.memory.load_memory_variables({})["history"]

    def setup_tools(self):
        self.faas_tool = FaasTool()
        self.tools = [self.faas_tool]

    def render_text_description_and_args(self, tools):
        # f.ex. network_retriever_json Useful for when you need to answer students questions about technology.Input should be a search query. {'query': 'search query to look up'},
        output = ''
        for tool in tools:
            formatted_args = {}
            for key, value in tool.args.items():
                formatted_args[key] = value['description']
            output += tool.name + ': ' + tool.description + ' ' + str(formatted_args) + ','
        return output
    
    def create_structured_chat_agent(self, llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate) -> Runnable:
        # I needed custom control over how tools are displayed in the prompt

        missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
            prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        prompt = prompt.partial(
            tools=self.render_text_description_and_args(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )
        llm_with_stop = llm.bind(stop=["Observation"])

        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | (lambda x: (print("Final Prompt to LLM:", x), x)[1])
            | llm_with_stop
            | JSONAgentOutputParser()
        )
        return agent

    def setup_agent(self):
        # See https://python.langchain.com/docs/modules/agents/quick_start for more info
        agent = self.create_structured_chat_agent(self.llm, self.tools, self.agent_prompt_template)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3, # force the agent to stop after one tool use
            # early_stopping_method='generate' # lets the agent generate one final answer after being forced to stop
        ).with_config(
            {"run_name": "Agent"}
        )

        self.agent = agent_executor

    def generate_response(self, question, callback=None):

        # Stream for Web UI: https://python.langchain.com/docs/modules/agents/how_to/streaming
        # If this function is being called from the WebUI we add a callback listener, so that we can render the internal state of the LLM in the UI

        response = None
        if callback != None:
            response = self.agent.invoke(
                {
                    "input": question,
                    "chat_history": self.load_memory({})
                },
                {"callbacks": [callback]}
            )
        else:
            response = self.agent.invoke({
                "input": question,
                "chat_history": self.load_memory({})
            })
        
        output = response['output']

        # Should the llm fail to generate a string, we extract it manually
        if isinstance(output, dict):
            output = next(iter(output.values()))
        if isinstance(output, str):
            # No processing needed
            output = output
    
        self.memory.save_context({ "input": question }, {"output": output})

        return output