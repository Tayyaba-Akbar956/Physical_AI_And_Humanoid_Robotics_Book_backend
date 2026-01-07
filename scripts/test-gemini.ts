
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
    console.error('Missing GEMINI_API_KEY in .env');
    process.exit(1);
}

async function main() {
    try {
        console.log('Fetching available models via REST API...');
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${GEMINI_API_KEY}`);

        if (!response.ok) {
            const text = await response.text();
            console.error(`REST API Error: ${response.status} ${response.statusText}`);
            console.error('Body:', text);
            process.exit(1);
        }

        const data = await response.json();
        const models = data.models || [];

        console.log(`Found ${models.length} models.`);

        // Filter for gemini models
        const geminiModels = models.filter((m: any) => m.name.includes('gemini'));

        console.log('Available Gemini Models:');
        geminiModels.forEach((m: any) => console.log(`- ${m.name}`));

        // Suggestion
        const optimal = geminiModels.find((m: any) => m.name.includes('flash'));
        if (optimal) {
            console.log(`\nRecommendation: Use '${optimal.name.replace('models/', '')}'`);
        }
    } catch (error) {
        console.error('Fatal error:', error);
    }
}

main();
