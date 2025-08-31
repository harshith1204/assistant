"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const dotenv_1 = __importDefault(require("dotenv"));
const pino_1 = __importDefault(require("pino"));
const research_1 = require("./routes/research");
dotenv_1.default.config();
const logger = (0, pino_1.default)({ level: process.env.LOG_LEVEL || 'info' });
const app = (0, express_1.default)();
app.use(express_1.default.json({ limit: '2mb' }));
app.get('/health', (_req, res) => {
    res.json({ ok: true, service: 'research-service' });
});
app.use('/research', (0, research_1.researchRouter)(logger));
const port = Number(process.env.PORT || 8080);
app.listen(port, () => {
    logger.info({ port }, 'research-service listening');
});
