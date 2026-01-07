
import { QdrantClient } from '@qdrant/js-client-rest';
import dotenv from 'dotenv';
import path from 'path';

// Load env variables
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

if (!QDRANT_URL || !QDRANT_API_KEY) {
    console.error('Missing QDRANT_URL or QDRANT_API_KEY in .env');
    process.exit(1);
}

const client = new QdrantClient({
    url: QDRANT_URL,
    apiKey: QDRANT_API_KEY,
});

async function main() {
    const COLLECTION_NAME = 'textbook_content_embeddings';

    console.log(`Checking collection ${COLLECTION_NAME}...`);

    try {
        const info = await client.getCollection(COLLECTION_NAME);
        console.log('Collection found:', info);

        console.log('Creating payload index for "module_id"...');

        await client.createPayloadIndex(COLLECTION_NAME, {
            field_name: 'module_id',
            field_schema: 'keyword',
        });

        console.log('Index creation initiated! This might take a few seconds.');

        // Also create for chapter_id just in case
        await client.createPayloadIndex(COLLECTION_NAME, {
            field_name: 'chapter_id',
            field_schema: 'keyword',
        });
        console.log('Index for chapter_id also initiated.');

        console.log('Done!');
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
