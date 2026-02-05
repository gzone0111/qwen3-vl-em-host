import os
# set cuda visible devices if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

# Import your classes
from models import Qwen3VLEmbedder

app = FastAPI()

# Global model instance
embedder = None

@app.on_event("startup")
async def load_models():
    global embedder
    # Adjust path to your local model
    EMBED_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-8B"
    
    print("Loading Embedding Model...")
    # Using the refactored class with 32k context default
    embedder = Qwen3VLEmbedder(model_name_or_path=EMBED_MODEL_PATH)
    print("Model Loaded!")

# Pydantic model that is permissive enough to accept Strings (SDK default) or Dicts (Manual request)
class EmbeddingRequest(BaseModel):
    input: Union[str, List[Any]] 
    instruction: Optional[str] = None
    model: Optional[str] = "qwen3-vl-embedding"

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    try:
        batch_inputs = []
        
        # 1. Normalize input to a list
        raw_input = request.input if isinstance(request.input, list) else [request.input]

        # 2. Process each item in the batch
        for item in raw_input:
            # Case A: Item is already a Dictionary (e.g. via requests/custom client)
            if isinstance(item, dict):
                batch_inputs.append(item)
            
            # Case B: Item is a String (Standard OpenAI SDK constraint)
            elif isinstance(item, str):
                # Try to decode JSON string back to Dict (The "Trick")
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        batch_inputs.append(parsed)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # If not JSON, treat as raw text or image URL heuristic
                clean_item = item.strip()
                if clean_item.startswith(('http:', 'https:', 'file:')) or \
                   clean_item.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    batch_inputs.append({"image": clean_item})
                else:
                    batch_inputs.append({"text": clean_item})

        # 3. Apply instruction if provided globally
        if request.instruction:
            for item in batch_inputs:
                # Only apply if the item didn't have its own instruction
                if "instruction" not in item:
                    item["instruction"] = request.instruction

        # 4. Generate Embeddings
        # The Qwen3VLEmbedder.process method accepts a list of dicts
        
        embeddings_tensor = embedder.process(batch_inputs)
        
        # 5. Format Response to match OpenAI Spec
        data_response = []
        for i, emb in enumerate(embeddings_tensor):
            data_response.append({
                "object": "embedding",
                "embedding": emb.tolist(),
                "index": i
            })

        return {
            "object": "list",
            "data": data_response,
            "model": request.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }

    except Exception as e:
        # Print full trace for debugging server-side
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8237)