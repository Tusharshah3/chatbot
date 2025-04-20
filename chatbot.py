import os
import tempfile
import requests
import warnings
from urllib.parse import urljoin
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Optional Google Search imports (new vs deprecated)
try:
    from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
except ImportError:
    warnings.warn("Using deprecated GoogleSearch imports. Consider installing langchain-google-community")
    from langchain_community.utilities import GoogleSearchAPIWrapper
    from langchain_community.tools.google_search.tool import GoogleSearchResults

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# ─── Load environment ─────────────────────────────────────────────────────────────
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID", "")

# ─── Helper functions ────────────────────────────────────────────────────────────
def extract_pdf_links(url, base_domain="https://www.manit.ac.in"):
    try:
        resp = requests.get(url, verify=False, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                links.append(href if href.startswith('http') else urljoin(base_domain, href))
        return links
    except Exception as e:
        print(f"[extract_pdf_links] {e}")
        return []

def load_pdf_content(pdf_url):
    try:
        resp = requests.get(pdf_url, verify=False, timeout=15)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            path = tmp.name
        loader = PyPDFLoader(path)
        docs = loader.load()
        os.unlink(path)
        print(f"Loaded PDF: {pdf_url}")
        return docs
    except Exception as e:
        print(f"[load_pdf_content] {e}")
        return []

def safe_load_webpage(url):
    try:
        loader = WebBaseLoader(url)
        session = requests.Session()
        session.verify = False
        loader.session = session
        docs = loader.load()
        print(f"Loaded webpage: {url}")
        return docs
    except Exception as e:
        print(f"[safe_load_webpage] {e}")
        return []

def load_local_pdf(path):
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"Loaded local PDF: {path}")
        return docs
    except Exception as e:
        print(f"[load_local_pdf] {e}")
        return []

# ─── Build your knowledge base ──────────────────────────────────────────────────
all_docs = []

# 1) Local PDF
if os.path.exists("syllabus.pdf"):
    all_docs += load_local_pdf("syllabus.pdf")

# 2) Web pages & PDFs
manit_urls = [
    "https://www.manit.ac.in/",
    "https://www.manit.ac.in/academics",
    "https://www.manit.ac.in/content/electrical-engineering",
    "https://www.manit.ac.in/academics/programs"
]
for url in manit_urls:
    all_docs += safe_load_webpage(url)
    for pdf in extract_pdf_links(url)[:2]:
        all_docs += load_pdf_content(pdf)

if not all_docs:
    print("⚠️  No docs loaded; fallback to live search only.")
else:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(all_docs)
    print(f"Document chunks: {len(documents)}")

# 3) Vector store & retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb   = FAISS.from_documents(documents, embeddings) if all_docs else None
retriever  = vectordb.as_retriever(search_kwargs={"k": 5}) if vectordb else None

# ─── Agent & tools setup ────────────────────────────────────────────────────────
# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0.1, top_p=0.9, max_output_tokens=2048
)

# Tools
tools = []
if retriever:
    retriever_tool = create_retriever_tool(
        retriever, 
        name="knowledge_base",
        description="Search MANIT documents and PDFs"
    )
    tools.append(retriever_tool)

# Google Search
search_wrapper = GoogleSearchAPIWrapper(k=5)
tools.append(GoogleSearchResults(api_wrapper=search_wrapper))

# arXiv
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=300)
tools.append(ArxivQueryRun(api_wrapper=arxiv_wrapper))

# Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent  = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    max_iterations=5, early_stopping_method="generate"
)

# Direct QA chain
qa_prompt = ChatPromptTemplate.from_template("""
Answer based only on the context below. Think step-by-step.
<context>
{context}
</context>
Question: {input}
""")
document_chain   = create_stuff_documents_chain(llm, qa_prompt)
retrieval_chain  = create_retrieval_chain(retriever, document_chain) if retriever else None

def process_query(query: str, use_agent: bool = True) -> str:
    if use_agent:
        out = agent_executor.invoke({"input": query})
        return out["output"]
    else:
        out = retrieval_chain.invoke({"input": query})
        return out["answer"]

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def handle_query():
    data = request.get_json(silent=True)
    if not data:
        return jsonify(status="error", message="Request body must be JSON"), 400

    msg = data.get("message")
    if not msg:
        return jsonify(status="error", message="Missing 'message' field"), 400

    use_agent = bool(data.get("use_agent", True))
    try:
        resp_text = process_query(msg, use_agent)
        return jsonify(
            status="success",
            message_received=msg,
            use_agent=use_agent,
            response=resp_text
        ), 200
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy"), 200

if __name__ == '__main__':
    # Make sure your environment variables are set, then:
    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 3000))
