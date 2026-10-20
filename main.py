from fastapi import FastAPI
from openai import OpenAI
import os 
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import uuid
from pydantic import BaseModel
app = FastAPI()
class DocumentModel(BaseModel):
    title: str
    body: str
    description: str
    created_by: str
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


@app.post("/ingest")
def ingest_doc(document: DocumentModel):
    doc_id = str(uuid.uuid4())
    title = document.title
    body = document.body
    description = document.description
    created_by = document.created_by
    combined_text = f"{title} {body} {description}"
    embedding_field = open_router_client.embeddings.create(
        model="nvidia/nemotron-3-embed-1b:free",
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