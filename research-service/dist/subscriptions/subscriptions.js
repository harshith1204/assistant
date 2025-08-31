"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.scheduleSubscription = scheduleSubscription;
const node_cron_1 = __importDefault(require("node-cron"));
function scheduleSubscription(input, _logger) {
    const id = Buffer.from(`${input.query}-${Date.now()}`).toString('base64url');
    const cronExpr = input.cadence === 'weekly' ? '0 8 * * 1' : '0 8 1 * *';
    node_cron_1.default.schedule(cronExpr, () => {
        // In production, call research run here and diff prior results.
    });
    const nextRunAt = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
    return { id, query: input.query, cadence: input.cadence, nextRunAt };
}
