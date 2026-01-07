
import { GoogleGenerativeAI } from '@google/generative-ai';
import { QdrantClient } from '@qdrant/js-client-rest';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

if (!GEMINI_API_KEY || !QDRANT_URL || !QDRANT_API_KEY) {
    console.error('Missing env');
    process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
const qdrant = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });
const COLLECTION_NAME = 'textbook_content_embeddings';

async function main() {
    const query = process.argv[2] || "What is Module 2 about?";
    console.log(`Querying: "${query}"`);

    // 1. Generate Embedding
    const result = await embeddingModel.embedContent(query);
    const vector = result.embedding.values;

    // 2. Search Qdrant
    console.log('Searching Qdrant...');
    const searchResult = await qdrant.search(COLLECTION_NAME, {
        vector: vector,
        limit: 5,
        with_payload: true
    });

    console.log(`Found ${searchResult.length} results.`);
    searchResult.forEach((res, i) => {
        console.log(`\n[${i + 1}] Score: ${res.score.toFixed(4)} | Module: ${res.payload?.module_id}`);
        const content = res.payload?.content as string || '';
        console.log(`Preview: ${content.substring(0, 150)}...`);
    });
}

main();
