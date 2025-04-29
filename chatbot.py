import os
import tempfile
import requests
import warnings
import logging
import gc
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
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain import hub

# Optional Google Search imports
try:
    from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
except ImportError:
    warnings.warn("Using deprecated GoogleSearch imports. Install langchain-google-community")
    from langchain_community.utilities import GoogleSearchAPIWrapper
    from langchain_community.tools.google_search.tool import GoogleSearchResults

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
os.environ.update({"GOOGLE_API_KEY": GOOGLE_API_KEY, "GOOGLE_CSE_ID": GOOGLE_CSE_ID})

# FAISS index path and embedding model
_INDEX_PATH = "faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# URLs to crawl at index build time
MANIT_URLS = [
#     "https://www.manit.ac.in/",
#     "https://www.manit.ac.in/academics",
#     "https://www.manit.ac.in/content/electrical-engineering",
#     "https://www.manit.ac.in/academics/programs"
]


def extract_pdf_links(url, base_domain="https://www.manit.ac.in"):
    try:
        resp = requests.get(url, verify=False, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        return [href if href.startswith('http') else urljoin(base_domain, href)
                for a in soup.select('a[href$=".pdf"]')
                for href in [a['href']]]
    except Exception as e:
        logging.error(f"[extract_pdf_links] {e}")
        return []


def load_pdf_content(pdf_url):
    try:
        resp = requests.get(pdf_url, verify=False, timeout=15)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            path = tmp.name
        docs = PyPDFLoader(path).load()
        os.unlink(path)
        logging.info(f"Loaded PDF: {pdf_url}")
        return docs
    except Exception as e:
        logging.error(f"[load_pdf_content] {e}")
        return []


def safe_load_webpage(url):
    try:
        loader = WebBaseLoader(url)
        loader.session = requests.Session()
        loader.session.verify = False
        docs = loader.load()
        logging.info(f"Loaded webpage: {url}")
        return docs
    except Exception as e:
        logging.error(f"[safe_load_webpage] {e}")
        return []


def load_local_pdf(path):
    try:
        docs = PyPDFLoader(path).load()
        logging.info(f"Loaded local PDF: {path}")
        return docs
    except Exception as e:
        logging.error(f"[load_local_pdf] {e}")
        return []

# ─── FAISS index management ────────────────────────────────────────────────────
_graph = None

def build_vectordb():
    docs = []
    if os.path.exists("syllabus.pdf"):
        docs += load_local_pdf("syllabus.pdf")
    for url in MANIT_URLS:
        docs += safe_load_webpage(url)
        for pdf in extract_pdf_links(url)[:2]:
            docs += load_pdf_content(pdf)

    if not docs:
        logging.info("No documents loaded; skipping FAISS index.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logging.info(f"Building FAISS index with {len(chunks)} document chunks...")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(_INDEX_PATH)

    # free memory
    del docs, chunks
    gc.collect()
    return db


def get_vectordb():
    global _graph
    if _graph is None:
        try:
            _graph = FAISS.load_local(_INDEX_PATH, embeddings)
            logging.info("Loaded FAISS index from disk.")
        except Exception:
            _graph = build_vectordb()
    return _graph

# ─── Global agent & tools (initialized once) ─────────────────────────────────
# Initialize FAISS and tools at import time (once per worker)
vectordb = get_vectordb()

tools = []
if vectordb:
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    from langchain.tools import Tool as LCITool
    tools.append(
        LCITool.from_function(
            func=retriever.get_relevant_documents,
            name="knowledge_base",
            description="Search MANIT documents and PDFs"
        )
    )

# External live tools
tools.append(GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(k=5)))
tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=300)))

# LLM + agent prompt
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, top_p=0.9, max_output_tokens=2048)
prompt = hub.pull("hwchase17/openai-functions-agent")

# Build the agent executor once
tagent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=tagent, tools=tools, verbose=False, max_iterations=5, early_stopping_method="generate")

# ─── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def handle_query():
    logging.info("Received request")
    data = request.get_json(silent=True)
    if not data:
        return jsonify(status="error", message="Request body must be JSON"), 400

    msg = data.get("message")
    if not msg:
        return jsonify(status="error", message="Missing 'message' field"), 400

    try:
        response = agent_executor.invoke({"input": msg})["output"]
        return jsonify(status="success", message_received=msg, response=response), 200
    except Exception as e:
        logging.error(f"[handle_query] {e}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy"), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 3000)))

