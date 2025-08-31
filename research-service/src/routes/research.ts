import { Router } from 'express';
import type { Logger } from 'pino';
import { z } from 'zod';
import { runResearch } from '../services/runResearch';
import { memoryStore } from '../storage/memoryStore';
import { crmClientFactory } from '../thirdparty/crmClient';
import { PlanJSON, ResearchBrief } from '../types/types';
import { scheduleSubscription } from '../subscriptions/subscriptions';

const RunSchema = z.object({
  query: z.string().min(3),
  scope: z.array(z.string()).optional(),
  geo: z.array(z.string()).optional(),
  freshness: z.union([z.string(), z.object({ months: z.number().int().positive() })]).optional(),
});

const SaveSchema = z.object({
  briefId: z.string().uuid(),
  crmRef: z
    .object({ leadId: z.string().uuid().optional(), taskId: z.string().uuid().optional() })
    .partial()
    .optional(),
  pmsRef: z
    .object({ projectId: z.string().optional(), pageId: z.string().optional() })
    .partial()
    .optional(),
  attachments: z
    .array(
      z.object({
        type: z.enum(['pdf', 'screenshot', 'other']).default('other'),
        url: z.string().url(),
        name: z.string().optional(),
      }),
    )
    .optional(),
});

const IdeasToPlanSchema = z.object({
  briefId: z.string().uuid(),
  selections: z.array(z.number().int().nonnegative()).min(1),
});

const SubscribeSchema = z.object({
  query: z.string().min(3),
  cadence: z.enum(['weekly', 'monthly']),
  geo: z.array(z.string()).optional(),
  scope: z.array(z.string()).optional(),
});

export const researchRouter = (logger: Logger) => {
  const r = Router();

  r.post('/run', async (req, res) => {
    const parsed = RunSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    try {
      const brief: ResearchBrief = await runResearch(parsed.data, logger);
      memoryStore.saveBrief(brief);
      return res.json(brief);
    } catch (err: any) {
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
    const brief = memoryStore.getBrief(briefId);
    if (!brief) {
      return res.status(404).json({ error: 'brief_not_found' });
    }
    try {
      const crm = crmClientFactory({
        baseUrl: process.env.CRM_BASE_URL || 'https://stage-api.simpo.ai/crm',
        bearerToken: process.env.CRM_BEARER_TOKEN,
      });

      let notesId: string | undefined;
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
    } catch (err: any) {
      logger.error({ err }, 'failed to save brief');
      return res.status(500).json({ error: 'save_failed' });
    }
  });

  r.post('/ideas-to-plan', async (req, res) => {
    const parsed = IdeasToPlanSchema.safeParse(req.body);
    if (!parsed.success) {
      return res.status(400).json({ error: parsed.error.flatten() });
    }
    const brief = memoryStore.getBrief(parsed.data.briefId);
    if (!brief) {
      return res.status(404).json({ error: 'brief_not_found' });
    }
    const selectedIdeas = parsed.data.selections
      .map(i => brief.ideas[i])
      .filter(Boolean);
    const plan: PlanJSON = {
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
    const sub = scheduleSubscription(parsed.data, logger);
    return res.json(sub);
  });

  return r;
};

