# ================================================================# ================================================================
# WEEK 1 — Understanding LangGraph with a 4-agent system
# Using Ollama (LLaMA running locally — no API key, free)
#
# Flow: Planner → Researcher → Summariser → Critic
# ================================================================

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# ── STEP 1: Define the STATE ─────────────────────────────────────
# Shared notepad every agent reads from and writes to

class ResearchState(TypedDict):
    question:  str        # the user's original question
    plan:      List[str]  # planner agent writes the plan here
    research:  str        # researcher agent writes findings here
    final:     str        # summariser's executive summary
    critique:  str        # critic agent's review and score

# ── STEP 2: Set up the LLM ───────────────────────────────────────
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# ── Agent 1 — The Planner ────────────────────────────────────────
def planner_agent(state: ResearchState) -> ResearchState:
    print("\n🧠 PLANNER AGENT is thinking...")
    print(f"   Question received: {state['question']}")

    prompt = f"""You are a research planner.

The user wants to research: {state['question']}

Break this into exactly 3 specific research tasks.
Format your response as a numbered list:
1. [first task]
2. [second task]
3. [third task]

Be specific and concise."""

    response = llm.invoke(prompt)

    lines = response.content.strip().split('\n')
    tasks = [line.strip() for line in lines if line.strip() and line[0].isdigit()]

    print(f"   ✅ Plan created with {len(tasks)} tasks")
    for task in tasks:
        print(f"      {task}")

    return {"plan": tasks}

# ── Agent 2 — The Researcher ─────────────────────────────────────
def researcher_agent(state: ResearchState) -> ResearchState:
    print("\n🔍 RESEARCHER AGENT is working...")
    print(f"   Reading plan: {len(state['plan'])} tasks to complete")

    plan_text = "\n".join(state['plan'])

    prompt = f"""You are a financial research analyst.

Original question: {state['question']}

Research plan to follow:
{plan_text}

Complete each research task and provide a comprehensive answer.
Structure your response clearly with findings for each task."""

    response = llm.invoke(prompt)

    print("   ✅ Research complete")

    return {"research": response.content}

# ── Agent 3 — The Summariser ─────────────────────────────────────
def summariser_agent(state: ResearchState) -> ResearchState:
    print("\n📝 SUMMARISER AGENT is writing final answer...")

    prompt = f"""You are a senior analyst writing an executive summary.

Original question: {state['question']}

Research findings:
{state['research']}

Write a clear, concise executive summary in 3-4 paragraphs.
End with a clear conclusion."""

    response = llm.invoke(prompt)

    print("   ✅ Summary complete")

    return {"final": response.content}

# ── Agent 4 — The Critic (NEW) ───────────────────────────────────
# Reviews the summary, scores it, suggests improvements
def critic_agent(state: ResearchState) -> ResearchState:
    print("\n⚖️  CRITIC AGENT is reviewing the summary...")

    prompt = f"""You are a senior editor reviewing a financial report.

Original question: {state['question']}

The report to review:
{state['final']}

Evaluate this report on:
1. Completeness — does it fully answer the question?
2. Clarity — is it easy to understand?
3. Actionability — does it give useful conclusions?

Give a score out of 10 and 2-3 specific suggestions for improvement.
Format:
SCORE: X/10
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]"""

    response = llm.invoke(prompt)

    print("   ✅ Review complete")

    return {"critique": response.content}

# ── STEP 6: Build the Graph ───────────────────────────────────────
print("🔧 Building agent graph...")

graph = StateGraph(ResearchState)

# Add all 4 agents as nodes
graph.add_node("planner",    planner_agent)
graph.add_node("researcher", researcher_agent)
graph.add_node("summariser", summariser_agent)
graph.add_node("critic",     critic_agent)

# Connect them: planner → researcher → summariser → critic → end
graph.set_entry_point("planner")
graph.add_edge("planner",    "researcher")
graph.add_edge("researcher", "summariser")
graph.add_edge("summariser", "critic")
graph.add_edge("critic",     END)

app = graph.compile()
print("✅ Graph compiled successfully")

# ── STEP 7: Run it ───────────────────────────────────────────────
print("\n" + "="*60)
print("MULTI-AGENT FINANCIAL RESEARCH SYSTEM")
print("="*60)

question = "What are the key factors to consider when investing in Microsoft stock?"

print(f"\n📌 Question: {question}")
print("\n" + "-"*60)

result = app.invoke({
    "question": question,
    "plan":     [],
    "research": "",
    "final":    "",
    "critique": ""
})

print("\n" + "="*60)
print("FINAL RESEARCH REPORT")
print("="*60)
print(result["final"])

print("\n" + "="*60)
print("CRITIC'S REVIEW")
print("="*60)
print(result["critique"])

print("\n" + "="*60)
print("WHAT JUST HAPPENED:")
print("="*60)
print(f"✅ Planner    → broke question into {len(result['plan'])} tasks")
print(f"✅ Researcher → completed all research tasks")
print(f"✅ Summariser → wrote executive summary")
print(f"✅ Critic     → reviewed and scored the report")
print(f"✅ State      → passed between all 4 agents automatically")
print("\nThis is LangGraph. 4 agents, 1 shared State, connected by edges.")