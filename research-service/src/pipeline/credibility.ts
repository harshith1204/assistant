import type { NormalizedDoc } from './normalizer';

function domainReputation(url: string): number {
  try {
    const u = new URL(url);
    const host = u.hostname;
    if (host.endsWith('.gov') || host.endsWith('.gov.in')) return 0.95;
    if (host.endsWith('.edu')) return 0.9;
    if (host.includes('mckinsey') || host.includes('bcg') || host.includes('gartner')) return 0.85;
    if (host.includes('economictimes') || host.includes('thehindu') || host.includes('medianama')) return 0.75;
  } catch {}
  return 0.5;
}

export function scoreCredibility(docs: NormalizedDoc[]): (NormalizedDoc & { score: number })[] {
  return docs
    .map(d => ({ ...d, score: domainReputation(d.url) }))
    .sort((a, b) => b.score - a.score);
}

