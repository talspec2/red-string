from llama_cpp import Llama
import json
import time

# 1. SETUP: Point this to your specific file name
# model_path = "./models/red-string.Q4_K_M.gguf" 
model_path = "./models/meta-llama-3.1-8b.Q8_0.gguf"

print(f"🕵️ Loading 'Conspiracy Board' Brain from: {model_path}...")

# 2. LOAD: Initialize the model on CPU
# n_ctx=2048:  Limits memory usage (standard for short text extraction)
# verbose=False: Hides the messy technical logs so you just see the result
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4, # Uses 4 CPU cores
    verbose=False
)

# 3. DEFINE THE TEST
# We use the specific format you likely trained on (Instruction -> Response)
# If your training data looked different, adjust this prompt template!
text_to_analyze = """The landscape of modern technology was fundamentally altered in the late 1990s and early 2000s by a series of high-stakes ventures led by Elon Musk. Before he became a household name for space exploration, Musk co-founded X.com, an online financial services company that later merged with Confinity to create PayPal. This merger brought the South African-born entrepreneur into close collaboration with other influential figures, including Peter Thiel and Max Levchin. Following the acquisition of PayPal by eBay in 2002 for $1.5 billion, the 'PayPal Mafia'—a group of former employees and founders—dispersed to create some of the most dominant firms in Silicon Valley.

Leveraging his newfound wealth, Musk turned his sights toward the stars, establishing Space Exploration Technologies Corp., more commonly known as SpaceX, in June 2002. Based in Hawthorne, California, the company aimed to reduce space transportation costs to enable the colonization of Mars. Simultaneously, he recognized the potential of sustainable energy in the automotive sector. Although Tesla Motors was originally incorporated in July 2003 by Martin Eberhard and Marc Tarpenning, Musk led the Series A round of investment in 2004 and joined the board of directors as its chairman.

By 2008, a year of immense financial pressure, he assumed the role of CEO at Tesla, a position he has held ever since. Under his guidance, the firm launched the Model S and expanded into solar energy through the acquisition of SolarCity, a company founded by his cousins, Lyndon and Peter Rive. Today, his influence spans from the launch pads of Cape Canaveral to the gigafactories of Nevada and Texas. As SpaceX continues its partnership with NASA for the Artemis program, the intricate web of companies and innovators linked to Musk serves as a prime example of the interconnected nature of the modern industrial era."""

prompt = f"""### Instruction:
Extract the relationship triples from the text below and output them as a JSON list.

### Input:
{text_to_analyze}

### Response:
"""
start_time = time.perf_counter()
# 4. RUN INFERENCE
print("\n🧠 Thinking... (This should happen in < 2 seconds)")


output = llm(
    prompt,
    max_tokens=256,     # Give it enough room to write the JSON
    stop=["###"],       # Stop generating if it tries to start a new instruction
    echo=False,         # Don't repeat the prompt in the output
)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# 5. RESULT
print("\n--- EXTRACTED DATA ---")
result_text = output['choices'][0]['text'].strip()
print(result_text)
print(f"\n⏱️ Inference completed in {elapsed_time:.2f} seconds.")

# Optional: Try to parse it to prove it's valid JSON
try:
    data = json.loads(result_text)
    print(f"\n✅ VALID JSON DETECTED: Found {len(data)} connections.")
except:
    print("\n⚠️ Raw text returned (formatting might be imperfect).")