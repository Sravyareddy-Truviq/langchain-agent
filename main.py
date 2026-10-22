"""
LangChain Banking Agent
- Connects to shared MCP server via streamable HTTP (SSE)
- Auth: X-Agent-Key header (LANGCHAIN_AGENT_KEY)
- Frontend JWT validated against Supabase
- Uses langchain-mcp-adapters for MCP tool loading
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import dns_patch  # noqa: F401

import os
import asyncio

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
AGENT_KEY = os.environ.get("LANGCHAIN_AGENT_KEY", "b2242c3f74c7e3ccad9fed2ac73b651d")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://huhdpxvsdxnsfsqxwpux.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

app = FastAPI(title="LangChain Banking Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# In-memory chat history per session
_chat_history: dict[str, list] = {}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip("\"'")
GROQ_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "openai/gpt-oss-120b").strip("\"'")
if "llama-3.3" in GROQ_MODEL:
    GROQ_MODEL = "openai/gpt-oss-120b"

# ── LLMs & Fallback ──────────────────────────────────────────────────────────
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

groq_llm = None
if GROQ_API_KEY:
    try:
        from langchain_groq import ChatGroq
        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
    except Exception:
        pass

# When Gemini suffers 503 capacity spikes, automatically fall back to Groq
llm_with_fallback = gemini_llm.with_fallbacks([groq_llm]) if groq_llm else gemini_llm

def get_llm(model_preference: str = "auto"):
    if model_preference == "groq" and groq_llm:
        return groq_llm
    if model_preference == "gemini":
        return gemini_llm
    return llm_with_fallback

SYSTEM_PROMPT = (
    "You are a professional banking assistant powered by LangChain. "
    "Help users manage their accounts: check balances, transfer funds, "
    "view transaction history, pay bills, open/close accounts, check loan "
    "eligibility, and convert currencies. Be concise and confirm before transfers."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# ── Supabase JWT ─────────────────────────────────────────────────────────────
async def verify_supabase_jwt(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1]
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY}
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Supabase token")
    return resp.json()


# ── Chat ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = "auto"  # "auto" (Gemini with Groq fallback), "gemini", or "groq"


@app.get("/health")
def health():
    return {"status": "ok", "agent": "langchain"}


# ── MCP tools + executors (loaded once, reused across requests) ───────────────
_tools: list | None = None
_executors: dict[str, AgentExecutor] = {}
_tools_lock = asyncio.Lock()


async def get_tools(refresh: bool = False) -> list:
    """Tool discovery opens an SSE connection, so do it once; tool calls open their own sessions."""
    global _tools
    async with _tools_lock:
        if _tools is None or refresh:
            client = MultiServerMCPClient(
                {
                    "banking": {
                        "url": f"{MCP_SERVER_URL}/mcp/sse",
                        "transport": "sse",
                        "headers": {"X-Agent-Key": AGENT_KEY},
                    }
                }
            )
            loaded = await client.get_tools()
            for t in loaded:
                t.handle_tool_error = True  # tool failures go back to the LLM instead of aborting the turn
            _tools = loaded
            _executors.clear()
        return _tools


def get_executor(key: str, llm, tools: list) -> AgentExecutor:
    if key not in _executors:
        _executors[key] = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=False)
    return _executors[key]


def _is_capacity_error(err_str: str) -> bool:
    return "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str


async def _run(executor: AgentExecutor, message: str, history: list) -> str:
    result = await executor.ainvoke({"input": message, "chat_history": history})
    return result.get("output", "I couldn't process that.")


@app.post("/chat")
async def chat(body: ChatRequest, user: dict = Depends(verify_supabase_jwt)):
    user_id = user.get("id", "anon")
    sid = f"{user_id}_{body.session_id}"
    history = _chat_history.setdefault(sid, [])

    try:
        tools = await get_tools()
    except Exception as e:
        print(f"[langchain] MCP server unavailable: {e}")
        return {"reply": "The banking tool server is currently unavailable. Please try again shortly."}

    primary_key = body.model if body.model in ("gemini", "groq") else "auto"
    primary = get_executor(primary_key, get_llm(body.model), tools)

    reply = None
    err_str = ""
    for attempt in range(2):
        try:
            reply = await _run(primary, body.message, history)
            break
        except Exception as e:
            err_str = str(e)
            if attempt == 0 and _is_capacity_error(err_str):
                await asyncio.sleep(1.5)
                continue
            break

    if reply is None and groq_llm and body.model != "gemini" and primary_key != "groq":
        try:
            print(f"[langchain] Error with primary LLM ({err_str[:40]}), falling back to Groq...")
            reply = await _run(get_executor("groq", groq_llm, tools), body.message, history)
        except Exception as fb_err:
            print(f"[langchain] Groq fallback error: {fb_err}")

    if reply is None:
        print(f"[langchain] chat failed: {err_str}")
        if _is_capacity_error(err_str):
            return {"reply": "Google Gemini is momentarily experiencing high demand (503). Please resend your message in a few seconds."}
        return {"reply": "Sorry, something went wrong processing your request. Please try again."}

    history.append(HumanMessage(content=body.message))
    history.append(AIMessage(content=reply))
    _chat_history[sid] = history[-20:]
    return {"reply": reply}
