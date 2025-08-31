import type { Finding } from '../types/types';
import type { NormalizedDoc } from './normalizer';

export function extractFacts(docs: Array<NormalizedDoc & { score: number }>): Finding[] {
  // Naive extraction: take first sentences as quotes. Replace with NER/regex rules in prod.
  return docs.slice(0, 8).map(d => {
    const firstLine = d.contentText.split(/\n|\.\s/).slice(0, 2).join('. ');
    return {
      title: d.title || d.url,
      summary: firstLine.slice(0, 280),
      evidence: [
        { quote: firstLine.slice(0, 200), url: d.url, published: d.published },
      ],
      confidence: Math.min(0.95, Math.max(0.4, d.score)),
      recency: 'unknown',
    };
  });
}

