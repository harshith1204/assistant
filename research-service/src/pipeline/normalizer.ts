import type { FetchedDoc } from './fetcher';

export type NormalizedDoc = FetchedDoc & { hash: string; clusterKey: string };

function simpleHash(text: string): string {
  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (h * 31 + text.charCodeAt(i)) | 0;
  }
  return String(h);
}

export function normalizeAndCluster(docs: FetchedDoc[]): NormalizedDoc[] {
  const dedup = new Map<string, NormalizedDoc>();
  for (const d of docs) {
    const hash = simpleHash(d.contentText.slice(0, 2000));
    const clusterKey = d.title?.toLowerCase().split(/\s+/).slice(0, 4).join('-') || 'misc';
    const nd: NormalizedDoc = { ...d, hash, clusterKey };
    if (!dedup.has(hash)) dedup.set(hash, nd);
  }
  return Array.from(dedup.values());
}

