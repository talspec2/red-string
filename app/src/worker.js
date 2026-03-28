import { pipeline } from '@xenova/transformers';

let extractor = null;

// Helper function to calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

self.addEventListener('message', async (event) => {
    const { type, payload, id } = event.data;

    if (type === 'INIT') {
        try {
            // Load the feature extraction pipeline
            extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                quantized: true // Use 8-bit quantized model for browser efficiency
            });
            self.postMessage({ type: 'STATUS', status: 'READY' });
        } catch (error) {
            self.postMessage({ type: 'ERROR', error: error.message });
        }
    }

    if (type === 'COMPUTE_SIMILARITY') {
        if (!extractor) {
            self.postMessage({ type: 'ERROR', error: 'Model not loaded yet' });
            return;
        }

        const { targetText, candidateText, candidateEmbedding } = payload;

        try {
            // Generate embedding for the target (new) text
            const targetOutput = await extractor(targetText, { pooling: 'mean', normalize: true });
            const targetVector = Array.from(targetOutput.data);

            let candidateVector = candidateEmbedding;

            // If the candidate doesn't have a cached embedding, generate one
            if (!candidateVector) {
                const candidateOutput = await extractor(candidateText, { pooling: 'mean', normalize: true });
                candidateVector = Array.from(candidateOutput.data);
            }

            // Calculate similarity
            const similarity = cosineSimilarity(targetVector, candidateVector);

            self.postMessage({
                type: 'SIMILARITY_RESULT',
                id: id,
                similarity: similarity,
                targetVector: targetVector,
                candidateVector: candidateVector
            });

        } catch (error) {
            self.postMessage({ type: 'ERROR', error: error.message, id: id });
        }
    }
});