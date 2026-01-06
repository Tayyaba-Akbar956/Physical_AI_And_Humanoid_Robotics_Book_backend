import { Router } from 'express';
import prisma from '../db/prisma.js';

const router = Router();

router.get('/', async (req, res) => {
    try {
        // Basic health check - can we query the DB?
        await prisma.$queryRaw`SELECT 1`;
        res.json({ status: 'healthy', database: 'connected' });
    } catch (error) {
        res.status(500).json({ status: 'unhealthy', error: 'Database connection failed' });
    }
});

export default router;
