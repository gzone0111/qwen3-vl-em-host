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
import base64
from io import BytesIO
from PIL import Image

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
        # 1. Normalize input to a list
        raw_input = request.input if isinstance(request.input, list) else [request.input]
        batch_inputs = []

        for item in raw_input:
            # Handle the JSON-string trick from OpenAI SDK
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except:
                    pass 

            # Prepare the dictionary for the model
            model_item = {}
            
            # Helper to normalize input into {"text": ..., "image": ...}
            # Handles raw strings (text/url) and dictionaries
            if isinstance(item, str):
                clean_item = item.strip()
                if clean_item.startswith(("http:", "https:")):
                    model_item["image"] = clean_item
                elif clean_item.startswith("data:image"): # Detect Base64 String
                    header, encoded = clean_item.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    model_item["image"] = Image.open(BytesIO(image_data)).convert("RGB")
                else:
                    model_item["text"] = item
            elif isinstance(item, dict):
                if "text" in item:
                    model_item["text"] = item["text"]
                if "image" in item:
                    img_val = item["image"]
                    # INTERCEPT BASE64 STRING HERE
                    if isinstance(img_val, str) and img_val.startswith("data:image"):
                        header, encoded = img_val.split(",", 1)
                        image_data = base64.b64decode(encoded)
                        # CONVERT TO PIL IMAGE OBJECT
                        model_item["image"] = Image.open(BytesIO(image_data)).convert("RGB")
                    else:
                        model_item["image"] = img_val # Keep URL/Path as string

            # Apply instruction if present
            if request.instruction and "instruction" not in model_item:
                model_item["instruction"] = request.instruction
                
            batch_inputs.append(model_item)

        # 2. Pass the list of dicts (now containing PIL objects) to the model
        embeddings = embedder.process(batch_inputs)
        
        # 3. Return response (Standard OpenAI Format)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": emb.tolist(), "index": i} 
                for i, emb in enumerate(embeddings)
            ],
            "model": request.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8237)