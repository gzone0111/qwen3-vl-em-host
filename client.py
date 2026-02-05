import json
from openai import OpenAI

# 1. Setup Client
client = OpenAI(
    base_url="http://localhost:8211/v1",
    api_key="sk-dummy" # Required by SDK, ignored by server
)

# 2. Your Multimodal Documents
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "/home/httsangaj/projects/qwen-vl-emb-host/AutoGraph-R1.png"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "/home/httsangaj/projects/qwen-vl-emb-host/demo.jpeg"}
]

queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "A graph showing the architecture of AutoGraph-R1"}
]

# 3. The "JSON String" Trick
# We convert the dicts to strings so they pass OpenAI SDK validation
input_strings = [json.dumps(doc) for doc in documents]
query_strings = [json.dumps(query) for query in queries]

emb_strings =  query_strings + input_strings
# 4. Call API
response = client.embeddings.create(
    model="qwen3-vl-embedding",
    input=emb_strings
)

# 5. Verify Results
print(f"Batch Size: {len(response.data)}") # Should be 6
embeddings = [item.embedding for item in response.data]
import numpy as np
embeddings_array = np.array(embeddings)
similarity_scores = (embeddings_array[:3]@ embeddings_array[3:].T )  # Query vs Document
print(similarity_scores.tolist())