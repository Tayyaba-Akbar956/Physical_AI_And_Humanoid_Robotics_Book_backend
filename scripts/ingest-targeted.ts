
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { QdrantClient } from '@qdrant/js-client-rest';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const DOCS_DIR = path.resolve(process.cwd(), '../docs');
const TARGET_FILTER = process.argv[2]; // e.g. "module-04"

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

if (!GEMINI_API_KEY || !QDRANT_URL || !QDRANT_API_KEY) {
    console.error('Missing env variables.');
    process.exit(1);
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
const qdrant = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });
const COLLECTION_NAME = 'textbook_content_embeddings';

function getFiles(dir: string, fileList: string[] = []) {
    const files = fs.readdirSync(dir);
    files.forEach(file => {
        const filePath = path.join(dir, file);
        if (fs.statSync(filePath).isDirectory()) {
            getFiles(filePath, fileList);
        } else {
            if (file.endsWith('.md') || file.endsWith('.mdx')) {
                fileList.push(filePath);
            }
        }
    });
    return fileList;
}

function chunkText(text: string, maxChars = 1000): string[] {
    const paragraphs = text.split(/\n\s*\n/);
    const chunks: string[] = [];
    let currentChunk = '';
    for (const para of paragraphs) {
        if ((currentChunk + para).length > maxChars) {
            if (currentChunk) chunks.push(currentChunk.trim());
            currentChunk = para;
        } else {
            currentChunk += (currentChunk ? '\n\n' : '') + para;
        }
    }
    if (currentChunk) chunks.push(currentChunk.trim());
    return chunks;
}

async function main() {
    if (!TARGET_FILTER) {
        console.error('Please provide a filter arg (e.g. module-04)');
        process.exit(1);
    }

    console.log(`Targeted Ingestion: ${TARGET_FILTER}`);
    console.log(`Scanning docs from: ${DOCS_DIR}`);

    const files = getFiles(DOCS_DIR);
    // Filter files
    const targetFiles = files.filter(f => f.includes(TARGET_FILTER));

    console.log(`Found ${targetFiles.length} markdown files matching "${TARGET_FILTER}".`);

    for (const file of targetFiles) {
        console.log(`Processing: ${path.relative(DOCS_DIR, file)}`);
        const content = fs.readFileSync(file, 'utf-8');
        const cleanContent = content.replace(/^---[\s\S]*?---\n/, '');
        const relPath = path.relative(DOCS_DIR, file);
        const parts = relPath.split(path.sep);
        const moduleId = parts[0].includes('module') ? parts[0] : 'general';
        const chapterId = path.basename(file, path.extname(file));
        const chunks = chunkText(cleanContent);
        const points = [];
        for (const chunk of chunks) {
            if (chunk.length < 50) continue;
            try {
                const result = await embeddingModel.embedContent(chunk);
                points.push({
                    id: uuidv4(),
                    vector: result.embedding.values,
                    payload: {
                        content: chunk,
                        module_id: moduleId,
                        chapter_id: chapterId,
                        file_path: relPath,
                        content_type: 'text'
                    }
                });
                // Rate limiting pause: 4000ms
                await new Promise(r => setTimeout(r, 4000));
            } catch (err: any) {
                console.error(`  Error embedding chunk: ${err.message}`);
                await new Promise(r => setTimeout(r, 10000));
            }
        }
        if (points.length > 0) {
            try {
                await qdrant.upsert(COLLECTION_NAME, { wait: true, points });
                console.log(`  Uploaded ${points.length} chunks.`);
            } catch (err: any) {
                console.error(`  Error uploading to Qdrant: ${err.message}`);
            }
        }
    }
    console.log('Targeted ingestion complete!');
}
main();
