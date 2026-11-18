"""
Nishikant's Resume Agent  —  a simple AI agent for the Product Intern role.

What it does
------------
Ask it anything about Nishikant, or paste a job description, and it will:
  1. search_resume   -> find the relevant experience / projects / skills
  2. match_job       -> score how well the resume fits a JD (and show gaps)
  3. product_ideas   -> pitch feature ideas for a digital-lending app (CreditSea)

How it works (the "agent" part)
-------------------------------
- With an LLM key (GROQ_API_KEY or OPENAI_API_KEY): the LLM decides which tool
  to call (function calling), reads the tool result, and writes the answer.
- Without a key: a simple keyword router picks the tool. Runs fully offline.

Run
---
    pip install openai          # only needed for LLM mode
    python agent.py
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# 1. KNOWLEDGE: the resume, as structured data
# ---------------------------------------------------------------------------
RESUME = {
    "name": "Nishikant Kumar",
    "contact": "nishikant3745@gmail.com | github.com/Nishikant090",
    "summary": (
        "Data-driven builder who sits between business teams and engineering: gathers "
        "requirements from operations and sales stakeholders, defines KPIs, and ships the "
        "dashboards, workflows and product features that act on them. Hands-on with SQL, "
        "Power BI, Excel and Python, plus full-stack product work on onboarding, role-based "
        "access and payment flows."
    ),
    "education": (
        "B.Tech in Data Science & AI, IIIT Naya Raipur (2023 - Present), SGPA 8.3. "
        "Coursework: DBMS, DSA, Probability & Statistics, Machine Learning, Computer Networks."
    ),
    "experience": [
        {
            "role": "AI & Data Analytics Intern",
            "company": "TRDP Happy World (Mario Biscuit & Rusk), New Delhi",
            "dates": "Jun 2026 - Present",
            "points": [
                "Gathers requirements from operations and sales teams; turns analysis into "
                "recommendations on pricing, route-to-market and distributor performance.",
                "Designed KPI dashboards for sales performance, SKU revenue, inventory turnover "
                "and retailer segmentation.",
                "Automated reporting with SQL pipelines on ERP data, cutting manual effort ~40%.",
                "Building demand-forecasting models to reduce overstock and stockouts.",
            ],
        },
        {
            "role": "Full Stack Developer Intern (Remote)",
            "company": "MillionMinds",
            "dates": "Jun 2026 - Present",
            "points": [
                "Building a learning platform end to end: onboarding, enrollment, progress "
                "tracking, quizzes and automated certificates.",
                "Shipped the course purchase flow with payment and communication APIs; "
                "JWT authentication and role-based permissions.",
                "Added lifecycle tracking, automated notifications and admin reports.",
            ],
        },
        {
            "role": "Web Developer Intern",
            "company": "Excel Group Pvt. Ltd., New Delhi",
            "dates": "May 2025 - Jul 2025",
            "points": [
                "Built responsive UI and landing pages (React, Vue, JavaScript) on PHP/MySQL backends.",
            ],
        },
    ],
    "projects": [
        {
            "name": "Customer Churn Prediction (End-to-End ML)",
            "detail": "Found retention drivers on 7,043 telco customers with SHAP; shipped a "
                      "Streamlit app with what-if analysis for non-technical users.",
        },
        {
            "name": "Cold Email Generator (GenAI Product)",
            "detail": "One-input product: paste a careers URL; LLaMA 3.1 + LangChain + ChromaDB "
                      "match the job to a portfolio and draft outreach.",
        },
        {
            "name": "VS Code Security Extension (Hack-a-sol 3.0 Winner)",
            "detail": "Real-time flags for OWASP Top 10 issues, hardcoded credentials and insecure APIs.",
        },
    ],
    "research": [
        "Quantum-resilient multi-hop communication with lattice-based cryptography "
        "(accepted at SPARCQC 2026, NIT Rourkela).",
        "Spark pipeline + stacked ensemble + 36-agent MARL for 5G handover prediction "
        "(ROC-AUC 0.9946, <6.1 ms latency).",
    ],
    "skills": [
        "requirement gathering", "kpi definition", "dashboarding", "stakeholder communication",
        "demand forecasting", "sql", "mysql", "postgresql", "power bi", "excel", "python",
        "pandas", "scikit-learn", "git", "streamlit", "rest api", "fastapi", "jwt", "rbac",
        "payment integration", "llm", "rag", "react", "machine learning", "data analysis",
    ],
    "achievements": [
        "Winner, Hack-a-sol 3.0 (50+ teams)",
        "Top 10, SMACKATHON 2024 (100+ teams, AI healthcare workflow platform)",
    ],
}


def _resume_chunks():
    """Flatten the resume into small searchable text chunks."""
    r = RESUME
    chunks = [("Summary", r["summary"]), ("Education", r["education"])]
    for e in r["experience"]:
        chunks.append((f"{e['role']} @ {e['company']} ({e['dates']})", " ".join(e["points"])))
    for p in r["projects"]:
        chunks.append((f"Project: {p['name']}", p["detail"]))
    for x in r["research"]:
        chunks.append(("Research", x))
    chunks.append(("Skills", ", ".join(r["skills"])))
    chunks.append(("Achievements", "; ".join(r["achievements"])))
    return chunks


STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "your",
             "you", "me", "my", "is", "are", "what", "tell", "about", "did", "do", "have", "any"}


# Recruiter words -> words the resume actually uses (added after eval.py caught misses)
SYNONYMS = {
    "study": {"education", "b.tech"}, "studying": {"education", "b.tech"},
    "college": {"education", "iiit"}, "university": {"education", "iiit"},
    "degree": {"education", "b.tech"},
    "cgpa": {"sgpa"}, "gpa": {"sgpa"}, "grades": {"sgpa"}, "marks": {"sgpa"},
    "hackathon": {"hack-a-sol", "smackathon", "winner"},
    "hackathons": {"hack-a-sol", "smackathon", "winner"},
    "llm": {"llama", "langchain", "genai", "rag"}, "llms": {"llama", "langchain", "genai", "rag"},
    "genai": {"llama", "langchain", "llm"}, "ai": {"genai", "llama", "machine"},
}


def _words(text):
    return set(re.findall(r"[a-z0-9+#/.-]+", text.lower())) - STOPWORDS


def _expand(words):
    """Add resume vocabulary for common recruiter phrasings."""
    out = set(words)
    for w in words:
        out |= SYNONYMS.get(w.strip("?.!,"), set())
    return out


# ---------------------------------------------------------------------------
# 2. TOOLS: plain Python functions the agent can call
# ---------------------------------------------------------------------------
def search_resume(query: str, top_k: int = 3) -> str:
    """Return the resume sections most related to the query (keyword overlap)."""
    q = _expand(_words(query))
    scored = []
    for title, text in _resume_chunks():
        score = len(q & _words(title + " " + text))
        if score:
            scored.append((score, title, text))
    scored.sort(reverse=True)
    if not scored:
        return "No direct match. Summary: " + RESUME["summary"]
    return "\n\n".join(f"[{t}]\n{x}" for _, t, x in scored[:top_k])


def match_job(job_description: str) -> str:
    """Score how well the resume fits a job description and list strengths and gaps."""
    jd = job_description.lower()
    skills = RESUME["skills"]
    matched = [s for s in skills if s in jd]
    # common product / analyst keywords to look for in the JD
    wanted = ["sql", "excel", "python", "power bi", "tableau", "jira", "figma", "a/b testing",
              "user research", "product analytics", "dashboarding", "stakeholder communication",
              "requirement gathering", "prd", "wireframe", "agile", "lending", "fintech",
              "kpi definition", "api"]
    asked = [w for w in wanted if w in jd]
    gaps = [w for w in asked if w not in skills and not any(w in s for s in skills)]
    covered = [w for w in asked if w not in gaps]
    score = round(100 * len(covered) / len(asked)) if asked else min(100, 40 + 10 * len(matched))
    return json.dumps({
        "fit_score_percent": score,
        "strengths_found_in_jd": sorted(set(matched + covered)),
        "gaps_to_address": gaps,
        "tip": "Lead with the TRDP stakeholder/KPI work and the MillionMinds onboarding + payment "
               "flow; they map directly to product ownership.",
    }, indent=2)


def product_ideas(product_area: str = "digital lending") -> str:
    """Suggest product features for a lending app, each tied to Nishikant's experience."""
    ideas = [
        ("Smarter onboarding funnel",
         "Track drop-off at each KYC step and test shorter flows. "
         "Built from: MillionMinds onboarding + lifecycle tracking."),
        ("Repayment-risk early warning",
         "A churn-style model that flags borrowers likely to miss an EMI, with SHAP reasons "
         "so ops know why. Built from: Customer Churn Prediction project."),
        ("Lender ops KPI dashboard",
         "Disbursals, approval rate, TAT, collections and NPA by segment in Power BI. "
         "Built from: TRDP KPI dashboards and SQL pipelines."),
        ("Personalised payment nudges",
         "Automated SMS/WhatsApp reminders timed by user behaviour. "
         "Built from: MillionMinds notifications + payment API work."),
        ("Fraud and security checks",
         "Flag risky sessions, devices and API misuse. "
         "Built from: VS Code Security Extension + cryptography research."),
    ]
    return f"Product ideas for {product_area}:\n" + "\n".join(
        f"{i}. {name}: {why}" for i, (name, why) in enumerate(ideas, 1))


TOOLS = {
    "search_resume": search_resume,
    "match_job": match_job,
    "product_ideas": product_ideas,
}

# Tool descriptions in the OpenAI function-calling format
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_resume",
        "description": "Search Nishikant's resume for experience, projects, skills or education.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "match_job",
        "description": "Score the resume against a pasted job description; returns strengths and gaps.",
        "parameters": {"type": "object", "properties": {
            "job_description": {"type": "string"}}, "required": ["job_description"]}}},
    {"type": "function", "function": {
        "name": "product_ideas",
        "description": "Suggest product features for a fintech / lending app, tied to his experience.",
        "parameters": {"type": "object", "properties": {
            "product_area": {"type": "string"}}, "required": []}}},
]

SYSTEM_PROMPT = (
    "You are Nishikant Kumar's resume agent, talking to a recruiter for a Product Intern role "
    "at a fintech (CreditSea, digital lending). Always use a tool to look up facts; never invent "
    "experience. Answer in 3-6 short, confident sentences or bullets, and mention real numbers "
    "from the resume where they help."
)


# ---------------------------------------------------------------------------
# 3. THE AGENT LOOP
# ---------------------------------------------------------------------------
def _llm_client():
    """Return (client, model) if an API key is set, else (None, None)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, None
    if os.getenv("GROQ_API_KEY"):
        return OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                      base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(), "gpt-4o-mini"
    return None, None


def run_llm_agent(client, model, history, user_msg, max_steps=4, tool_log=None):
    """LLM picks tools -> we run them -> LLM writes the final answer.

    If tool_log is a list, the name of every tool called is appended to it
    (used by eval.py to check the agent picked the right tool).
    """
    history.append({"role": "user", "content": user_msg})
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=history, tools=TOOL_SCHEMAS, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:                       # no more tools needed -> done
            history.append({"role": "assistant", "content": msg.content})
            return msg.content
        history.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:                  # run every tool the LLM asked for
            args = json.loads(call.function.arguments or "{}")
            print(f"   [tool] {call.function.name}({', '.join(args)})")
            if tool_log is not None:
                tool_log.append(call.function.name)
            result = TOOLS[call.function.name](**args)
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Sorry, I couldn't finish that. Try rephrasing."


def run_offline_agent(user_msg):
    """No API key: pick a tool with simple rules."""
    tool, result = route_offline(user_msg)
    print(f"   [tool] {tool}")
    return result


def route_offline(user_msg):
    """Keyword router. Returns (tool_name, tool_result)."""
    text = user_msg.lower()
    if len(user_msg) > 250 or any(k in text for k in ["job description", "jd:", "requirements", "responsibilities"]):
        tool, result = "match_job", match_job(user_msg)
    elif any(k in text for k in ["idea", "feature", "improve", "creditsea", "lending", "suggest"]):
        tool, result = "product_ideas", product_ideas()
    elif any(k in text for k in ["why", "hire", "pitch", "yourself", "who are"]):
        tool = "search_resume"
        result = search_resume("requirements stakeholders kpi dashboards onboarding payment product", top_k=3)
    else:
        tool, result = "search_resume", search_resume(user_msg)
    return tool, result


def main():
    client, model = _llm_client()
    mode = f"LLM mode ({model})" if client else "offline mode (set GROQ_API_KEY for LLM answers)"
    print("=" * 64)
    print(f" {RESUME['name']} - Resume Agent | {mode}")
    print(" Try: 'Why should we hire you?', 'Tell me about your SQL work',")
    print("      'Product ideas for CreditSea', or paste a job description.")
    print(" Type 'quit' to exit.")
    print("=" * 64)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            break
        if client:
            answer = run_llm_agent(client, model, history, user_msg)
        else:
            answer = run_offline_agent(user_msg)
        print(f"\nAgent: {answer}")
    print("\nThanks! Contact: " + RESUME["contact"])


if __name__ == "__main__":
    main()
