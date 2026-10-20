from fastapi import FastAPI
from openai import OpenAI
import os 
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import uuid
from pydantic import BaseModel
from laya import Router
import logging

logger = logging.getLogger("uvicorn.error")
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
        response = open_router_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Use the following context to answer the query:\n\nContext:\n{context}\n\nQuery: {query}"}
            ]
        )
        
        answer = response.choices[0].message.content
        return {"answer": answer, "sources": search_results}
    else:
        return hybrid_search(SearchRequest(query=query))
    



router = Router(preload=True)

classifier_questions = {
    "should_we_call_llm": {
        "type": "noul",
        "instructions": "Does this user query require calling a large language model? Return true ONLY for genuine questions, complex requests, or queries needing explanation. Return false for greetings (hello, hi, how are you), single words, simple acknowledgements, or anything that is not a real question."
    }
}


def classify_query(query: str) -> bool:
    state_context = {
        "user_query": query,
        "instruction": (
            "Decide whether this user query requires calling a large language model "
            "to generate a meaningful answer. Return true if the query is a real question, "
            "a complex request, or needs explanation. Return false if it is a single word, "
            "a greeting, a simple acknowledgement, or does not need an AI-generated answer."
        )
    }
    results = router.predict(state=state_context, questions=classifier_questions)
    noul_score = results.get("answers", {}).get("should_we_call_llm", {}).get("noul", 1.0)
    logging.warning(f"Query: {query}, noul_score: {noul_score}")
    return noul_score >= 0.80
