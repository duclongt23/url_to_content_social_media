from state import (
    InputState,
    SummaryOutputState,
    ResearchOutputState, 
    IntentMatchingInputState,
    FinalState,
    GeneratedContent,
    SummaryInputState
)
from prompt import (summary_prompt, 
                    research_agent_prompt, 
                    instagram_prompt,
                    linkedin_prompt,
                    twitter_prompt,
                    blog_prompt)

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from newspaper import Article
from langgraph.graph import StateGraph, START, END


load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "social-media-content-generation"
os.getenv("TAVILY_API_KEY")

summ_model = ChatOpenAI(model = "gpt-4.1-mini", temperature=0.4)

model = ChatOpenAI(model = "gpt-4.1-mini", temperature=0.6)

research_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

class ResearchQuestion(TypedDict):
    questions: list[str]

#node
def parsingArticle(state: InputState) -> SummaryInputState:
    print("***** Parsing the given URL *****")
    article = Article(state["url"])
    article.download()
    article.parse()
    text = article.text
    platforms = state["platforms"]
    
    return {
        "text": text,
        "platforms": platforms
    }


def summary_text(state: SummaryInputState) -> SummaryOutputState:
    print("***** Generating summary of the given text *****")
    summary = summ_model.invoke(summary_prompt.invoke({
        "text": state["text"]
    })).content
    
    return {
        "text": state["text"],
        "text_summary": summary,
        "platforms": state["platforms"]
    }

def research_node(state: SummaryOutputState) -> ResearchOutputState:
    print("***** Researching for the best content *****")
    res = model.with_structured_output(ResearchQuestion, strict = True).invoke(research_agent_prompt.invoke({
        "text_summary": state["text_summary"],
        "platforms": state["platforms"]
    }))
    response = research_tool.batch(res["questions"])
    research = ""

    for i, ques in enumerate(res["questions"]):
        research += f"Question : {ques}\n"
        research += "Answer " + "\n\n".join([res['content'] for res in response[i]]) + "\n\n"

    return {
        "text": state["text"],
        "research": research,
        "platforms": state["platforms"]
    }

def intent_matching(state: ResearchOutputState):
    print("******* Sending data to each Platfrom *************")
    # platform_nodes = []
    # for platform in state["platforms"]:
    #     platform_nodes.append(Send(platform, {"text": state["text"],"research": state["research"], "platform": platform}))
    # return platform_nodes

def Insta(state: IntentMatchingInputState) ->  FinalState:
    if not "Instagram" in state["platforms"]:
        return {"contents": [""]}
    res = model.invoke(instagram_prompt.invoke({
        "text": state["text"],
        "research": state["research"]
    }))
    return {"contents": ["Insta\n" + res.content]}

def Twitter(state: IntentMatchingInputState) -> FinalState:
    if not "Twitter" in state["platforms"]:
        return {"contetns": [""]}
    res = model.invoke(twitter_prompt.invoke({
        "text": state["text"],
        "research": state["research"]
    }))
    return {"contents": ["Twitter\n" + res.content]}

def Linkedin(state: IntentMatchingInputState) -> FinalState:
    if not "Linkedin" in state["platforms"]:
        return {"contents": [""]}
    res = model.invoke(linkedin_prompt.invoke({
        "text": state["text"],
        "research": state["research"]
    }))
    return { "contents": ["Linkedin\n" + res.content]}

def Blog(state: IntentMatchingInputState) -> FinalState:
    if not "Blog" in state["platforms"]:
        return {"contents": [""]}
    res = model.invoke(blog_prompt.invoke({
        "text": state["text"],
        "research": state["research"]
    }))
    return {"contents": ["Blog\n" + res.content]}

def combining_content(state: FinalState) -> GeneratedContent:
    final_content = ""
    for content in state["contents"]:
        final_content +=  content + "\n******************************\n"
    return {"generated_content": final_content}   

#graph
builder = StateGraph(input= InputState, output= GeneratedContent)
builder.add_node("parsingArticle", parsingArticle)
builder.add_node("summary_node", summary_text)
builder.add_node("research_node", research_node)
builder.add_node("intent_matching", intent_matching)
builder.add_node("instagram", Insta)
builder.add_node("twitter", Twitter)
builder.add_node("linkedin", Linkedin)
builder.add_node("blog", Blog)
builder.add_node("combine_content", combining_content)

#Flow
builder.add_edge(START, "parsingArticle")
builder.add_edge("parsingArticle", "summary_node")
builder.add_edge("summary_node", "research_node")
builder.add_edge("research_node", "intent_matching")
builder.add_edge("intent_matching", "instagram")
builder.add_edge("intent_matching", "twitter")
builder.add_edge("intent_matching", "linkedin")
builder.add_edge("intent_matching","blog")
builder.add_edge("instagram", "combine_content")
builder.add_edge("twitter", "combine_content")
builder.add_edge("linkedin", "combine_content")
builder.add_edge("blog", "combine_content")
builder.add_edge("combine_content", END)

graph = builder.compile()

with open("graph.png", "wb") as f:
    f.write((graph.get_graph().draw_mermaid_png()))

url = input("Enter the URL: ")
res = graph.invoke({"url": url, "platforms": ["Blog", "Linkedin"]})
with open("example.md", "w", encoding="utf-8") as f:
    f.write(res["generated_content"])
print(res["generated_content"])