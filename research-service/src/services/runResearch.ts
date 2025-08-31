import type { Logger } from 'pino';
import { v4 as uuid } from 'uuid';
import { ResearchBrief, Finding, Idea } from '../types/types';
import { expandQueries } from '../pipeline/queryUnderstanding';
import { searchWeb } from '../pipeline/search';
import { fetchAndExtract } from '../pipeline/fetcher';
import { normalizeAndCluster } from '../pipeline/normalizer';
import { scoreCredibility } from '../pipeline/credibility';
import { extractFacts } from '../pipeline/extraction';
import { summarizeBrief } from '../pipeline/summarize';
import { reviewBrief } from '../pipeline/review';

type RunInput = {
  query: string;
  scope?: string[];
  geo?: string[];
  freshness?: string | { months: number };
};

export async function runResearch(input: RunInput, logger: Logger): Promise<ResearchBrief> {
  const now = new Date();
  const briefId = uuid();
  const entities = expandQueries.extractEntities(input.query);
  const keyQuestions = expandQueries.expandKeyQuestions(input.query);
  const subQueries = expandQueries.expandSubQueries(input.query, input.geo, input.scope);

  const searchResults = await searchWeb(subQueries, input.freshness, logger);
  const fetched = await fetchAndExtract(searchResults, logger);
  const normalized = normalizeAndCluster(fetched);
  const scored = scoreCredibility(normalized);
  const topDocs = scored.slice(0, 20);
  const findings: Finding[] = extractFacts(topDocs);

  const ideas: Idea[] = summarizeBrief.generateIdeas(findings, input.query);
  const summary = await summarizeBrief.summarize(findings, input.query);
  const reviewed = reviewBrief({ findings, summary });

  const brief: ResearchBrief = {
    briefId,
    query: input.query,
    date: now.toISOString(),
    entities,
    keyQuestions,
    findings: reviewed.findings,
    ideas,
    attachments: [],
    summary: reviewed.summary,
  };
  logger.info({ briefId, findings: brief.findings.length }, 'research complete');
  return brief;
}

