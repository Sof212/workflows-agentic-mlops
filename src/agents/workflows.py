
from typing import TypedDict, List,Literal,Optional
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from IPython.display import Image, display

# 1. Define State
class State(TypedDict):

    # User query
    query: str

    # Router decision
    route: Literal["RAG", "PYTHON", "NOTHING"]

    # Python execution
    requirements: List[str]
    python_code: str
    python_correct: bool
    python_error: Optional[str]

    # RAG response validation
    reponse_check: bool
    documents: List[Document]

    # Final response
    reponse_final: str


# 2. Define Workflow
class WorkflowLangGraph:

    def __init__(
        self,
        query: str,
    ):

        self.query = query
        self.agents_verify_code=agents_verify_code()
        self.coder=CodeGeneration()
        self.router=RouterAgent()
        self.retriever = RetrieverBuilder()
        self.research_agent = ResearchAgent(
            query=self.query,
        )

        self.workflow = self.build_workflow()

    # 3. Build Graph
    def build_workflow(self):

        workflow = StateGraph(State)

        # Nodes
        workflow.add_node("agent_router", self._router_agent)
        workflow.add_node("agent_checker", self._agent_checker)
        workflow.add_node("coder", self._agent_coder)

        # Entry point
        workflow.set_entry_point("agent_router")

        # Conditional routing
        workflow.add_conditional_edges(
            "agent_router",
            self._conditional_edge,
            {
                "agent_checker": "agent_checker",
                "coder": "coder",
                END: END
            }
        )

        # Endpoints
        workflow.add_edge("coder", END)
        workflow.add_edge("agent_checker", END)

        return workflow.compile()

        # 4. Agent Checker
    def _router_agent(self, state: State):
            query = state["query"]

            prompt = self.router.generate_prompt(query)
            route = self.router.generate_finale_reponses(prompt)

            return {"route": route}


    def _agent_coder(self, state: State):
        rep = self.coder.generate_code(state["query"])

        return {
            "python_code": rep.python_code,
            "requirements": rep.requirements
        }


    def _agent_checker(self, state: State):
        query = state["query"]

        retriever = self.retriever.build_hybrid_retriever(
            k=2,
            weights=[0.5, 0.5]
        )

        results = retriever.invoke(query)

        response = self.research_agent.generate_answer(
            query=query,
            documents=results
        )

        return {
            "reponse_check": response.Boolean,
            "documents": results,
            "reponse_final": response.responses
        }

         
    def display_graph(self):
            graph = self.workflow.get_graph()
            display(Image(graph.draw_mermaid_png()))
    def _conditional_edge(
        self,
        state: State
    ) -> Literal["agent_checker", "coder", END]:

        route = state["route"]

        if route == "RAG":
            return "agent_checker"

        elif route == "PYTHON":
            return "coder"

        return END
    def run(self):

        initial_state: State = {
            "query": self.query,
            "route": "NOTHING",
            "requirements": [],
            "python_code": "",
            "python_correct": False,
            "python_error": None,
            "reponse_check": False,
            "documents": [],
            "reponse_final": ""
        }

        return self.workflow.invoke(initial_state)
