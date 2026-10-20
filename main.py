from fastapi import FastAPI
from openai import OpenAI
import os 
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import uuid
from pydantic import BaseModel
from laya import Router
import logging

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
