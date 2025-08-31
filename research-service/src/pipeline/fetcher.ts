import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import type { Logger } from 'pino';
import type { SearchHit } from './search';

export type FetchedDoc = {
  url: string;
  title?: string;
  contentText: string;
  published?: string;
};

async function fetchOne(hit: SearchHit, logger: Logger): Promise<FetchedDoc | null> {
  try {
    const resp = await fetch(hit.url, { method: 'GET' });
    if (!resp.ok) return null;
    const html = await resp.text();
    const dom = new JSDOM(html, { url: hit.url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();
    const contentText = article?.textContent?.trim() || '';
    if (!contentText) return null;
    return {
      url: hit.url,
      title: article?.title || hit.title,
      contentText,
      published: hit.published,
    };
  } catch (err) {
    logger.debug({ url: hit.url, err: String(err) }, 'fetch failed');
    return null;
  }
}

export async function fetchAndExtract(hits: SearchHit[], logger: Logger): Promise<FetchedDoc[]> {
  const tasks = hits.slice(0, 20).map(h => fetchOne(h, logger));
  const results = await Promise.all(tasks);
  return results.filter((r): r is FetchedDoc => Boolean(r));
}

