"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.researchRouter = void 0;
const express_1 = require("express");
const zod_1 = require("zod");
const runResearch_1 = require("../services/runResearch");
const memoryStore_1 = require("../storage/memoryStore");
const crmClient_1 = require("../thirdparty/crmClient");
const subscriptions_1 = require("../subscriptions/subscriptions");
const RunSchema = zod_1.z.object({
    query: zod_1.z.string().min(3),
    scope: zod_1.z.array(zod_1.z.string()).optional(),
    geo: zod_1.z.array(zod_1.z.string()).optional(),
    freshness: zod_1.z.union([zod_1.z.string(), zod_1.z.object({ months: zod_1.z.number().int().positive() })]).optional(),
});
const SaveSchema = zod_1.z.object({
    briefId: zod_1.z.string().uuid(),
    crmRef: zod_1.z
        .object({ leadId: zod_1.z.string().uuid().optional(), taskId: zod_1.z.string().uuid().optional() })
        .partial()
        .optional(),
    pmsRef: zod_1.z
        .object({ projectId: zod_1.z.string().optional(), pageId: zod_1.z.string().optional() })
        .partial()
        .optional(),
    attachments: zod_1.z
        .array(zod_1.z.object({
        type: zod_1.z.enum(['pdf', 'screenshot', 'other']).default('other'),
        url: zod_1.z.string().url(),
        name: zod_1.z.string().optional(),
    }))
        .optional(),
});
const IdeasToPlanSchema = zod_1.z.object({
    briefId: zod_1.z.string().uuid(),
    selections: zod_1.z.array(zod_1.z.number().int().nonnegative()).min(1),
});
const SubscribeSchema = zod_1.z.object({
    query: zod_1.z.string().min(3),
    cadence: zod_1.z.enum(['weekly', 'monthly']),
    geo: zod_1.z.array(zod_1.z.string()).optional(),
    scope: zod_1.z.array(zod_1.z.string()).optional(),
});
const researchRouter = (logger) => {
    const r = (0, express_1.Router)();
    r.post('/run', async (req, res) => {
        const parsed = RunSchema.safeParse(req.body);
        if (!parsed.success) {
            return res.status(400).json({ error: parsed.error.flatten() });
        }
        try {
            const brief = await (0, runResearch_1.runResearch)(parsed.data, logger);
            memoryStore_1.memoryStore.saveBrief(brief);
            return res.json(brief);
        }
        catch (err) {
            logger.error({ err }, 'failed to run research');
            return res.status(500).json({ error: 'research_failed' });
        }
    });
    r.post('/save', async (req, res) => {
        const parsed = SaveSchema.safeParse(req.body);
        if (!parsed.success) {
            return res.status(400).json({ error: parsed.error.flatten() });
        }
        const { briefId, crmRef, attachments } = parsed.data;
        const brief = memoryStore_1.memoryStore.getBrief(briefId);
        if (!brief) {
            return res.status(404).json({ error: 'brief_not_found' });
        }
        try {
            const crm = (0, crmClient_1.crmClientFactory)({
                baseUrl: process.env.CRM_BASE_URL || 'https://stage-api.simpo.ai/crm',
                bearerToken: process.env.CRM_BEARER_TOKEN,
            });
            let notesId;
            if (crmRef) {
                const noteResp = await crm.createNote({
                    subject: `Research Brief: ${brief.query} (${brief.date})`,
                    description: JSON.stringify({ briefId: brief.briefId, summary: brief.summary }),
                    leadId: crmRef.leadId,
                });
                notesId = noteResp.notesId;
                if (crmRef.taskId && attachments && attachments.length > 0) {
                    await crm.createTaskAttachment(crmRef.taskId, {
                        title: 'Research artifacts',
                        description: `Artifacts for brief ${brief.briefId}`,
                        attachmentUrl: attachments.map(a => ({ url: a.url, name: a.name })),
                    });
                }
            }
            return res.json({ saved: true, crmNotesId: notesId });
        }
        catch (err) {
            logger.error({ err }, 'failed to save brief');
            return res.status(500).json({ error: 'save_failed' });
        }
    });
    r.post('/ideas-to-plan', async (req, res) => {
        const parsed = IdeasToPlanSchema.safeParse(req.body);
        if (!parsed.success) {
            return res.status(400).json({ error: parsed.error.flatten() });
        }
        const brief = memoryStore_1.memoryStore.getBrief(parsed.data.briefId);
        if (!brief) {
            return res.status(404).json({ error: 'brief_not_found' });
        }
        const selectedIdeas = parsed.data.selections
            .map(i => brief.ideas[i])
            .filter(Boolean);
        const plan = {
            okrs: [
                {
                    objective: `Execute research-driven initiatives for: ${brief.query}`,
                    keyResults: [
                        {
                            metric: 'validated-ideas',
                            target: selectedIdeas.length,
                        },
                    ],
                },
            ],
            initiatives: selectedIdeas.map((idea, idx) => ({
                id: `init-${idx + 1}`,
                title: idea.idea,
                tasks: [
                    { id: `task-${idx + 1}-1`, title: 'Define success metrics', assignee: 'TBD' },
                    { id: `task-${idx + 1}-2`, title: 'Run pilot/experiment', assignee: 'TBD' },
                    { id: `task-${idx + 1}-3`, title: 'Analyze results', assignee: 'TBD' },
                ],
            })),
        };
        return res.json(plan);
    });
    r.post('/subscribe', async (req, res) => {
        const parsed = SubscribeSchema.safeParse(req.body);
        if (!parsed.success) {
            return res.status(400).json({ error: parsed.error.flatten() });
        }
        const sub = (0, subscriptions_1.scheduleSubscription)(parsed.data, logger);
        return res.json(sub);
    });
    return r;
};
exports.researchRouter = researchRouter;
