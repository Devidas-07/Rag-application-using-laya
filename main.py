from fastapi import FastAPI
from openai import OpenAI
import os 
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import uuid
from pydantic import BaseModel
from laya import Router
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
app = FastAPI()
class DocumentModel(BaseModel):
    title: str
    body: str
    description: str
    created_by: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5
@app.get("/")
def get_health():
    return "app health is ok"

load_dotenv()


try:
    base_url = os.getenv("base_url")
    api_key = os.getenv("api_key")
    open_router_client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    )
    print(f"connected to openrouter with {base_url} and {api_key}")
except Exception as e:
    print(f"Error creating OpenAI client: {e}")
try:
    es_host = os.getenv("es_host")
    es_client = Elasticsearch(hosts=es_host)
    es_index = os.getenv("es_index")
    print(f"connected to elasticsearch with {es_host}")
except Exception as e:
    print(f"Error creating Elasticsearch client: {e}")

try:
    embedding_model = os.getenv("embedding_model")
    print(f"using embedding model {embedding_model}")
except Exception as e:
    print(f"Error setting embedding model: {e}")

try:
    llm_model = os.getenv("llm_model")
    print(f"using llm model {llm_model}")
except Exception as e:
    print(f"Error setting llm model: {e}")

@app.post("/ingest")
def ingest_doc(document: DocumentModel):
    doc_id = str(uuid.uuid4())
    title = document.title
    body = document.body
    description = document.description
    created_by = document.created_by
    combined_text = f"{title} {body} {description}"
    embedding_field = open_router_client.embeddings.create(
        model=embedding_model,
        input=combined_text,
        encoding_format="float"
    )
    es_payload = {
        "id": doc_id,
        "title": title,
        "body": body,
        "description": description,
        "created_by": created_by,
        "embedding_field": embedding_field.data[0].embedding
    }
    result = es_client.index(index=es_index, id=doc_id, 
        document=es_payload)
    return result
@app.post("/bulk-ingest")
def bulk_ingest_docs(documents: list[DocumentModel]):
    ids = [str(uuid.uuid4()) for _ in documents]
    combined_texts = [f"{d.title} {d.body} {d.description}" for d in documents]

    embeddings_response = open_router_client.embeddings.create(
        model=embedding_model,
        input=combined_texts,
        encoding_format="float"
    )
    embeddings = [e.embedding for e in embeddings_response.data]

    bulk_body = []
    for doc_id, doc, embedding in zip(ids, documents, embeddings):
        bulk_body.append({"index": {"_index": es_index, "_id": doc_id}})
        bulk_body.append({
            "id": doc_id, "title": doc.title, "body": doc.body,
            "description": doc.description, "created_by": doc.created_by,
            "embedding_field": embedding
        })

    result = es_client.bulk(body=bulk_body)
    return {"status": "success", "count": len(documents), "errors": result["errors"]}

@app.post("/hybrid-search")
def hybrid_search(request_data: SearchRequest):
    query = request_data.query
    k = request_data.k
    # Generate embedding for the query
    query_embedding = open_router_client.embeddings.create(
        model=embedding_model,
        input=query,
        encoding_format="float"
    )
    
    # Define the hybrid search query
    search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title","body","description","created_by"],
                                "fuzziness": "AUTO",
                                "boost": 1.0
                            }
                        }
                    ],
                    "minimum_should_match": 0
                }
            },
            "knn": {
                "field": "embedding_field",
                "query_vector": query_embedding.data[0].embedding,
                "k": k,
                "num_candidates": 100,
                "boost": 2.0
            }
        }
    
    # Execute the search
    response = es_client.search(index=es_index, body=search_body)
    
    # Extract and return the results
    results = []
    for hit in response['hits']['hits']:
        results.append({
            "id": hit['_id'],
            "title": hit['_source'].get('title', ''),
            "description": hit['_source'].get('description', ''),
            "body": hit['_source'].get('body', ''),
            "created_by": hit['_source'].get('created_by', ''),
            "score": hit['_score']
        })
    
    return results

@app.post("/convesational-search")
def generate_answer(request_data: SearchRequest):
    query = request_data.query
    k = request_data.k
    #check cache
    cached_result = check_cache(query)
    if cached_result:
        logging.warning("cache result found")
        return cached_result
    #laya will check do we need to call llm or not?
    should_call_llm = classify_query(query)
    logging.warning(f"Query: {query}, Should call LLM: {should_call_llm}")
    if should_call_llm:
        # Perform hybrid search
        search_results = hybrid_search(SearchRequest(query=query, k=5))
        
        # Format the context from search results
        context = "\n".join([
            f"Title: {r['title']}\nDescription: {r['description']}\nBody: {r['body']}\n"
            for r in search_results
        ])
        
        # Generate answer using LLM
        #prompt builder check guardrails
        harmful_content = is_harmful_input(query)
        if harmful_content:
            return {"sorry we can't poceed with harmful query"}
        else:
            logging.warning(f"No harmful content")
            response = open_router_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Use the following context to answer the query:\n\nContext:\n{context}\n\nQuery: {query} give answer in only 5 to 6 lines. If you don't have context then directly say I don't have enough information so I can not generate answer"}
                ]
            )
        
        answer = response.choices[0].message.content
        if answer is not None:
            logging.warning(f"saving response in cache")
            save_to_cache(query, answer, search_results)
            return {"answer": answer, "sources": search_results}
        else:
            return {"error": "Failed to generate an answer."}
    else:
        logging.warning(f"calling hybrid search")
        return hybrid_search(SearchRequest(query=query))
    



router = Router(preload=True)

classifier_questions = {
    "should_we_call_llm": {
        "type": "noul",
        "instructions": "Does this user query require calling a large language model? Return true ONLY for genuine questions, complex requests, or queries needing explanation. Return false for greetings (hello, hi, how are you), single words, simple acknowledgements, or anything that is not a real question."
    }
}

is_harmful_question = {
    "is_harful_query": {
        
        "type": "noul",
        "instructions": "Is this query potentially harmful, offensive, or outside the scope of helpful responses? Return true for any query that could be harmful, inappropriate, or unrelated to the intended use case. does it contain any forbidden content?"

    }
}

# Call the LLM unless the classifier is fairly sure it is NOT needed.
LLM_THRESHOLD = 0.35          # tune on your own logged data
MAX_QUERY_CHARS = 1500        # Laya's context is limited, so truncate

# Messages that are purely social: full-match only, so "hi, what is your refund policy?" is NOT skipped.
_SOCIAL_ONLY = re.compile(
    r"^\s*(hi+|hello+|hey+|hii+|yo|namaste|namaskar|good\s+(morning|afternoon|evening|night)|"
    r"thanks?(\s+you)?|thank\s*u|thx|ok(ay)?|k|cool|great|nice|got\s*it|sure|"
    r"bye|goodbye|see\s+you|"
    r"[\U0001F44D\U0001F64F\U0001F60A\U0001F642\u2705]+)"
    r"[\s!.,?]*$",
    re.IGNORECASE,
)

_QUESTION_START = re.compile(
    r"^\s*(what|why|how|when|where|who|which|whom|can|could|should|would|will|do|does|did|"
    r"is|are|was|were|explain|describe|compare|summari[sz]e|list|show|give|tell|help|"
    r"write|create|generate|translate|fix|debug|"
    r"kya|kaise|kyun|kab|kahan|kaun|kitna)\b",
    re.IGNORECASE,
)

GATE_INSTRUCTION = (
    "You are a gate in front of a document-based assistant. DEFAULT TO TRUE. "
    "Answer true if the message can be answered with knowledge, documents or reasoning. "
    "This includes: questions with or without a question mark; short keyword or topic lookups "
    "(for example 'refund policy', 'pricing', 'reset password'); commands such as "
    "'summarize', 'explain', 'compare', 'write', 'translate'; troubleshooting and error messages; "
    "follow-up messages like 'why?', 'tell me more' or 'and for yearly plan?' when a previous "
    "assistant message is present; and any non-English or mixed-language query (Hindi, Hinglish). "
    "Answer false ONLY when the message is purely social or has no informational content: "
    "greetings, thanks, acknowledgements, farewells, empty text, gibberish, or only punctuation/emojis. "
    "If you are unsure, answer true."
)


def classify_query(query: str, previous_assistant_message: Optional[str] = None) -> bool:
    """Return True if the query should be sent to the LLM / RAG pipeline."""
    q = (query or "").strip()

    # 1. Deterministic rules (free, fast, predictable)
    if not q:
        return False
    if _SOCIAL_ONLY.match(q):
        logger.info("Query: %r -> skip LLM (rule: social only)", q)
        return False
    if "?" in q or _QUESTION_START.match(q) or len(q.split()) >= 6:
        logger.info("Query: %r -> call LLM (rule: clear question/request)", q)
        return True

    # 2. Ambiguous cases go to the decision model
    state_context = {
        "user_query": q[:MAX_QUERY_CHARS],
        "previous_assistant_message": (previous_assistant_message or "")[:500],
        "instruction": GATE_INSTRUCTION,
    }

    try:
        results = router.predict(state=state_context, questions=classifier_questions)
        noul_score = (
            results.get("answers", {})
            .get("should_we_call_llm", {})
            .get("noul", 1.0)   # fail-open: if the score is missing, call the LLM
        )
    except Exception:
        logger.exception("Classifier failed for query %r, failing open", q)
        return True              # never block a user because the gate crashed

    decision = noul_score >= LLM_THRESHOLD
    logger.info("Query: %r, noul_score: %.3f, threshold: %.2f, call_llm: %s",
                q, noul_score, LLM_THRESHOLD, decision)
    return decision
def is_harmful_input(query:str) -> bool:
    #check guardrails using laya decision model
    state_context = {
        "user_query": query,
        "instruction": (
           "Decide whether this user query contain any harmful content"
        )
    }
    results = router.predict(state=state_context, questions=classifier_questions)
    noul_score = results.get("answers", {}).get("should_we_call_llm", {}).get("noul", 1.0)
    logging.warning(f"Query: {query}, noul_score: {noul_score}")
    return noul_score >= 0.80

def check_cache(query: str):
    cache_index = os.getenv("es_cache_index")
    search_body = {
        "query": {
            "term": {
                "query.keyword": {"value": query}
            }
        }
    }
    es_result = es_client.search(index=cache_index, body=search_body)
    hits = es_result['hits']['hits']
    if not hits or '_source' not in hits[0]:
        return None
    source = hits[0]['_source']
    answer = source.get('answer')
    if not answer:
        return None
    return {"answer": answer, "sources": source.get('sources', [])}

def save_to_cache(query: str, answer: str, sources: list):
    cache_index = os.getenv("es_cache_index")
    doc_id = str(uuid.uuid4())
    es_payload = {
        "id": doc_id,
        "query": query,
        "answer": answer,
        "sources": sources
    }
    result = es_client.index(index=cache_index, id=doc_id, document=es_payload)
    return result

