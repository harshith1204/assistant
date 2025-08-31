import type { Logger } from 'pino';

export type SearchHit = { url: string; title?: string; snippet?: string; published?: string };

export async function searchWeb(queries: string[], _freshness: unknown, logger: Logger): Promise<SearchHit[]> {
  // Placeholder: integrate with Bing/Google/News via env keys.
  // For now, return unique URLs guessed from queries to keep flow working in dev.
  const hits: SearchHit[] = [];
  const seen = new Set<string>();
  for (const q of queries) {
    const fakeUrl = `https://www.example.com/search?q=${encodeURIComponent(q)}`;
    if (!seen.has(fakeUrl)) {
      seen.add(fakeUrl);
      hits.push({ url: fakeUrl, title: q });
    }
  }
  logger.debug({ count: hits.length }, 'search results');
  return hits.slice(0, 50);
}

