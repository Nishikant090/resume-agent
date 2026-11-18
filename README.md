# Nishikant's Resume Agent

A small AI agent that answers recruiter questions about me, using my resume as its only source of facts. Built for the **Product Intern** role at CreditSea.

## What it can do
| Tool | What it does |
|---|---|
| `search_resume` | Finds the relevant experience, projects or skills for a question |
| `match_job` | Scores my resume against a pasted job description and lists the gaps |
| `product_ideas` | Suggests features for a digital-lending app, each linked to work I've done |

## How it works
1. The user asks a question.
2. The agent decides which tool to call. With an API key the LLM chooses through function calling; without one, a simple keyword router chooses.
3. The tool runs on the resume data, and the agent writes the answer from what the tool returns.

The resume is stored as structured data, and the LLM is told to use a tool for every fact, so it can't invent experience I don't have.

## Run
```bash
python agent.py                      # offline mode, no key needed

pip install openai
export GROQ_API_KEY=your_key         # free key at console.groq.com (LLaMA 3.1)
python agent.py                      # LLM mode
```

## Run with Docker
```bash
docker build -t resume-agent .
docker run -it --rm resume-agent                              # offline mode
docker run -it --rm -e GROQ_API_KEY=your_key resume-agent     # LLM mode
docker run --rm resume-agent python eval.py                   # run the evals
```

## Evals
`eval.py` runs 17 test questions from [`evals/questions.json`](evals/questions.json) through the agent and checks three things:
- **Tool choice:** did the agent call the right tool?
- **Grounding:** does the answer contain the real facts (e.g. "40%", "7,043", "8.3")?
- **Hallucination traps:** questions like *"Has he worked at Google?"* that the LLM must answer "no" to, not make up.

```bash
python eval.py          # offline router
python eval.py --llm    # LLM agent (needs an API key)
```
It prints pass rate, tool accuracy and latency, saves `evals/results.json`, and exits with code 1 if the pass rate is below the threshold (default 80%), so it can gate CI.

**What it caught:** the first run scored **13/17 (76%)**. Keyword search missed recruiter phrasings like "CGPA" (resume says SGPA), "hackathons" (resume says Hack-a-sol) and "LLMs" (resume says LLaMA). Adding a small synonym map brought it to **17/17 (100%)** with 100% tool accuracy.

## Try asking
- Why should we hire you?
- Tell me about your SQL work
- Product ideas for CreditSea
- *(paste a full job description)*

## Product thinking
- **User:** a recruiter with 2 minutes to decide.
- **Job to be done:** "Is this candidate a fit for my role?"
- **Success metric:** the recruiter gets a grounded answer in a single question.
- **Next steps:** a Streamlit chat UI, answers that cite the resume line they came from, and logging which questions recruiters ask most.
