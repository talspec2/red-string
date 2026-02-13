import re
import json
import time
from llama_cpp import Llama

# 1. SETUP
model_path = "./models/meta-llama-3.1-8b.Q8_0.gguf"
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)

def extract_with_sliding_window(full_text):
    # Improved splitter: handles decimals like 1.5 without breaking
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    sentences = [s for s in sentences if len(s.split()) > 2] # Filter junk
    
    all_triplets = []
    seen_triplets = set()

    print(f"🔬 Investigation started: {len(sentences)} valid sentences found.")
    start_time = time.time()
    
    for i in range(len(sentences) - 1):
        window_text = f"{sentences[i]} {sentences[i+1]}"
        print(f"\n--- Window {i+1}/{len(sentences)-1} ---")
        print(f"Text: \"{window_text[:60]}...\"")
        
        # We REMOVE the [ from the Response line to see if the model starts it naturally
        prompt = f"### Instruction:\nExtract relationship triples as a JSON list. If no relations exist, return [].\n\n### Input:\n{window_text}\n\n### Response:\n"
        
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["###"] # Remove "]" from stop to let it finish the list
        )

        raw_output = output['choices'][0]['text'].strip()
        print(f"Model Output: {raw_output}") # DEBUG: See exactly what it said

        try:
            # Clean potential lead-in text (like "Here is the JSON:")
            json_start = raw_output.find("[")
            json_end = raw_output.rfind("]") + 1
            
            if json_start != -1 and json_end != -1:
                clean_json = raw_output[json_start:json_end]
                data = json.loads(clean_json)
                
                for triple in data:
                    # Deduplication
                    key = (triple.get('head', '').lower().strip(), 
                           triple.get('type', '').lower().strip(), 
                           triple.get('tail', '').lower().strip())

                    if key not in seen_triplets and all(key):
                        seen_triplets.add(key)
                        all_triplets.append(triple)
                        print(f"  ✅ Added: {triple['head']} -> {triple['type']}")
            else:
                print("  ⚠️ No JSON brackets found in output.")
        except Exception as e:
            print(f"  ❌ Parse Error: {e}")

    total_time = time.time() - start_time
    return all_triplets, total_time

# --- TEST ---
long_text = """The landscape of modern technology was fundamentally altered in the late 1990s and early 2000s by a series of high-stakes ventures led by Elon Musk. Before he became a household name for space exploration, Musk co-founded X.com, an online financial services company that later merged with Confinity to create PayPal. This merger brought the South African-born entrepreneur into close collaboration with other influential figures, including Peter Thiel and Max Levchin. Following the acquisition of PayPal by eBay in 2002 for $1.5 billion, the 'PayPal Mafia'—a group of former employees and founders—dispersed to create some of the most dominant firms in Silicon Valley.

Leveraging his newfound wealth, Musk turned his sights toward the stars, establishing Space Exploration Technologies Corp., more commonly known as SpaceX, in June 2002. Based in Hawthorne, California, the company aimed to reduce space transportation costs to enable the colonization of Mars. Simultaneously, he recognized the potential of sustainable energy in the automotive sector. Although Tesla Motors was originally incorporated in July 2003 by Martin Eberhard and Marc Tarpenning, Musk led the Series A round of investment in 2004 and joined the board of directors as its chairman.

By 2008, a year of immense financial pressure, he assumed the role of CEO at Tesla, a position he has held ever since. Under his guidance, the firm launched the Model S and expanded into solar energy through the acquisition of SolarCity, a company founded by his cousins, Lyndon and Peter Rive. Today, his influence spans from the launch pads of Cape Canaveral to the gigafactories of Nevada and Texas. As SpaceX continues its partnership with NASA for the Artemis program, the intricate web of companies and innovators linked to Musk serves as a prime example of the interconnected nature of the modern industrial era."""

final_data, duration = extract_with_sliding_window(long_text)
print(f"\n⏱️ Total Time: {duration:.2f}s")
print(f"📊 Final Triples Found: {len(final_data)}")
print(json.dumps(final_data, indent=2))