import { pipeline, cos_sim, env } from '@huggingface/transformers';

// 1. Force WebAssembly to load from the CDN to prevent Vite from returning HTML
// env.backends.onnx.wasm.numThreads = 1;

// 2. Disable local model checks and force remote Hugging Face Hub fetching
env.allowLocalModels = false;
env.allowRemoteModels = true;

// 3. Disable caching to force active network requests
env.useBrowserCache = false;

class EmbeddingPipeline {
    static task = 'feature-extraction';
    static model = 'Xenova/all-MiniLM-L6-v2';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { 
                progress_callback,
            });
        }
        return this.instance;
    }
}

self.addEventListener('message', async (event) => {
    const { triplets, threshold = 0.85 } = event.data;

    if (!triplets || triplets.length === 0) {
        self.postMessage({ status: 'complete', result: [] });
        return;
    }

    const validTriplets = triplets.filter(t => t && t.head && t.type && t.tail);
    if (validTriplets.length === 0) {
        self.postMessage({ status: 'complete', result: [] });
        return;
    }

    const tripletStrs = validTriplets.map(t => `head: ${t.head} | type: ${t.type} | tail: ${t.tail}`);
    
    // Post progress back to the main thread to verify network activity
    const extractor = await EmbeddingPipeline.getInstance((data) => {
        self.postMessage({ status: 'progress', data });
    });

    const output = await extractor(tripletStrs, { pooling: 'mean', normalize: true });
    const embeddings = output.tolist();

    const uniqueTriplets = [];
    const seenIndices = new Set();

    for (let i = 0; i < embeddings.length; i++) {
        if (seenIndices.has(i)) continue;
        uniqueTriplets.push(validTriplets[i]);
        seenIndices.add(i);

        for (let j = i + 1; j < embeddings.length; j++) {
            const similarity = cos_sim(embeddings[i], embeddings[j]);
            if (similarity >= threshold) {
                seenIndices.add(j);
            }
        }
    }

    self.postMessage({ status: 'complete', result: uniqueTriplets });
});
