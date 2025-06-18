import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import base64
from fastapi.responses import JSONResponse
import uvicorn
import traceback
from dotenv import load_dotenv
import sys
import pathlib

# Add parent directory to path to import embedding module
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from embedding import semantic_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base.db")
SIMILARITY_THRESHOLD = 0.68  # Lowered threshold for better recall
MAX_RESULTS = 10  # Increased to get more context
load_dotenv()
print("DEBUG: AIPROXY_TOKEN from os.environ =", os.getenv("AIPROXY_TOKEN"))
MAX_CONTEXT_CHUNKS = 4  # Increased number of chunks per source
API_KEY = os.getenv("AIPROXY_TOKEN")  # Get API key from environment variable

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify API key is set
if not API_KEY:
    logger.error("AIPROXY_TOKEN environment variable is not set. The application will not function correctly.")

# Create a connection to the SQLite database
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        error_msg = f"Database connection error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Make sure database exists or create it
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Create discourse_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER,
        topic_id INTEGER,
        topic_title TEXT,
        post_number INTEGER,
        author TEXT,
        created_at TEXT,
        likes INTEGER,
        chunk_index INTEGER,
        content TEXT,
        url TEXT,
        embedding BLOB
    )
    ''')
    
    # Create markdown_chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

# Vector similarity calculation with improved handling
def cosine_similarity(vec1, vec2):
    try:
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        logger.error(traceback.format_exc())
        return 0.0  # Return 0 similarity on error rather than crashing

# Function to get embedding from AIproxy
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        error_msg = "AIPROXY_TOKEN environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Getting embedding for text (length: {len(text)})")
            # Call AIproxy's embedding API
            url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            logger.info("Sending request to AIproxy embedding API")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received embedding")
                        return result["data"][0]["embedding"]  # Return the embedding vector
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error getting embedding (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception getting embedding (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to find similar content in the database with improved logic
async def find_similar_content(query_embedding, conn):
    try:
        logger.info("Finding similar content in database")
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        logger.info("Querying discourse chunks")
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        discourse_chunks = cursor.fetchall()
        logger.info(f"Found {len(discourse_chunks)} discourse chunks with embeddings")
        
        for chunk in discourse_chunks:
            try:
                chunk_embedding = np.frombuffer(chunk['embedding'], dtype=np.float32)
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        'id': chunk['id'],
                        'post_id': chunk['post_id'],
                        'topic_id': chunk['topic_id'],
                        'topic_title': chunk['topic_title'],
                        'post_number': chunk['post_number'],
                        'author': chunk['author'],
                        'created_at': chunk['created_at'],
                        'likes': chunk['likes'],
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'url': chunk['url'],
                        'similarity': similarity,
                        'source': 'discourse'
                    })
            except Exception as e:
                logger.warning(f"Error processing discourse chunk {chunk['id']}: {e}")
                continue
        
        # Search markdown chunks
        logger.info("Querying markdown chunks")
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        """)
        
        markdown_chunks = cursor.fetchall()
        logger.info(f"Found {len(markdown_chunks)} markdown chunks with embeddings")
        
        for chunk in markdown_chunks:
            try:
                chunk_embedding = np.frombuffer(chunk['embedding'], dtype=np.float32)
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        'id': chunk['id'],
                        'doc_title': chunk['doc_title'],
                        'original_url': chunk['original_url'],
                        'downloaded_at': chunk['downloaded_at'],
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'similarity': similarity,
                        'source': 'markdown'
                    })
            except Exception as e:
                logger.warning(f"Error processing markdown chunk {chunk['id']}: {e}")
                continue
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:MAX_RESULTS]
        
        logger.info(f"Returning {len(results)} similar results")
        return results
        
    except Exception as e:
        error_msg = f"Error finding similar content: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Function to enrich results with adjacent chunks for better context
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info("Enriching results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []
        
        for result in results:
            enriched_chunks = [result]
            
            if result['source'] == 'discourse':
                # Get adjacent chunks from the same post
                cursor.execute("""
                SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
                       likes, chunk_index, content, url, embedding 
                FROM discourse_chunks 
                WHERE post_id = ? AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """, (result['post_id'], 
                      max(0, result['chunk_index'] - 1), 
                      result['chunk_index'] + 1))
                
                adjacent_chunks = cursor.fetchall()
                for chunk in adjacent_chunks:
                    if chunk['id'] != result['id']:
                        enriched_chunks.append({
                            'id': chunk['id'],
                            'post_id': chunk['post_id'],
                            'topic_id': chunk['topic_id'],
                            'topic_title': chunk['topic_title'],
                            'post_number': chunk['post_number'],
                            'author': chunk['author'],
                            'created_at': chunk['created_at'],
                            'likes': chunk['likes'],
                            'chunk_index': chunk['chunk_index'],
                            'content': chunk['content'],
                            'url': chunk['url'],
                            'similarity': result['similarity'] * 0.8,  # Slightly lower similarity for adjacent chunks
                            'source': 'discourse'
                        })
            
            elif result['source'] == 'markdown':
                # Get adjacent chunks from the same document
                cursor.execute("""
                SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
                FROM markdown_chunks 
                WHERE doc_title = ? AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """, (result['doc_title'], 
                      max(0, result['chunk_index'] - 1), 
                      result['chunk_index'] + 1))
                
                adjacent_chunks = cursor.fetchall()
                for chunk in adjacent_chunks:
                    if chunk['id'] != result['id']:
                        enriched_chunks.append({
                            'id': chunk['id'],
                            'doc_title': chunk['doc_title'],
                            'original_url': chunk['original_url'],
                            'downloaded_at': chunk['downloaded_at'],
                            'chunk_index': chunk['chunk_index'],
                            'content': chunk['content'],
                            'similarity': result['similarity'] * 0.8,  # Slightly lower similarity for adjacent chunks
                            'source': 'markdown'
                        })
            
            # Limit the number of chunks per source
            enriched_chunks = enriched_chunks[:MAX_CONTEXT_CHUNKS]
            enriched_results.extend(enriched_chunks)
        
        # Remove duplicates and sort by similarity
        seen_ids = set()
        unique_results = []
        for result in enriched_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        logger.info(f"Enriched results: {len(unique_results)} unique chunks")
        return unique_results
        
    except Exception as e:
        error_msg = f"Error enriching with adjacent chunks: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return results  # Return original results if enrichment fails

# Function to generate answer using AIproxy
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        error_msg = "AIPROXY_TOKEN environment variable not set"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Prepare context from relevant results
    context_parts = []
    for result in relevant_results:
        if result['source'] == 'discourse':
            context_parts.append(f"From Discourse post '{result['topic_title']}' by {result['author']}:\n{result['content']}")
        elif result['source'] == 'markdown':
            context_parts.append(f"From document '{result['doc_title']}':\n{result['content']}")
    
    context = "\n\n".join(context_parts)
    
    # Create the prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from a knowledge base. 

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Be helpful and informative.

Answer:"""
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info("Generating answer using AIproxy")
            url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result["choices"][0]["message"]["content"].strip()
                        logger.info("Successfully generated answer")
                        return answer
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))  # Exponential backoff
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
        except Exception as e:
            error_msg = f"Exception generating answer (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(3 * retries)  # Wait before retry

# Function to get relevant links from results
def get_relevant_links(question: str) -> List[LinkInfo]:
    # This is a placeholder - you can implement more sophisticated link extraction
    # For now, we'll return some example links
    return [
        LinkInfo(url="https://example.com/doc1", text="Documentation 1"),
        LinkInfo(url="https://example.com/doc2", text="Documentation 2")
    ]

# Main query endpoint
@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.question}")
        
        # Get database connection
        conn = get_db_connection()
        
        # Get embedding for the question
        query_embedding = await get_embedding(request.question)
        
        # Find similar content
        similar_results = await find_similar_content(query_embedding, conn)
        
        if not similar_results:
            conn.close()
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or ask about a different topic.",
                links=[]
            )
        
        # Enrich with adjacent chunks
        enriched_results = await enrich_with_adjacent_chunks(conn, similar_results)
        
        # Generate answer
        answer = await generate_answer(request.question, enriched_results)
        
        # Get relevant links (you can implement this based on your needs)
        links = get_relevant_links(request.question)
        
        conn.close()
        
        return QueryResponse(answer=answer, links=links)
        
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "TDS Virtual TA API is running"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "TDS Virtual TA API", "docs": "/docs"}

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 