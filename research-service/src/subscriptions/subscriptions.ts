import cron from 'node-cron';
import type { Logger } from 'pino';

type Subscription = { id: string; query: string; cadence: 'weekly' | 'monthly'; nextRunAt: string };

export function scheduleSubscription(input: { query: string; cadence: 'weekly' | 'monthly' }, _logger: Logger): Subscription {
  const id = Buffer.from(`${input.query}-${Date.now()}`).toString('base64url');
  const cronExpr = input.cadence === 'weekly' ? '0 8 * * 1' : '0 8 1 * *';
  cron.schedule(cronExpr, () => {
    // In production, call research run here and diff prior results.
  });
  const nextRunAt = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
  return { id, query: input.query, cadence: input.cadence, nextRunAt };
}

