import express from 'express';
import dotenv from 'dotenv';
import pino from 'pino';
import { researchRouter } from './routes/research';

dotenv.config();

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });
const app = express();

app.use(express.json({ limit: '2mb' }));

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'research-service' });
});

app.use('/research', researchRouter(logger));

const port = Number(process.env.PORT || 8080);
app.listen(port, () => {
  logger.info({ port }, 'research-service listening');
});

