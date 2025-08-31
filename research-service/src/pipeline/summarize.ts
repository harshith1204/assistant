import type { Finding, Idea } from '../types/types';

async function summarize(findings: Finding[], query: string): Promise<string> {
  const bullets = findings.slice(0, 5).map(f => `- ${f.title}: ${f.summary}`);
  return [`Brief for: ${query}`, ...bullets].join('\n');
}

function generateIdeas(findings: Finding[], query: string): Idea[] {
  const base: Idea[] = [
    { idea: `Validate top 2 channels inferred from findings for: ${query}`, RICE: { reach: 1000, impact: 2, confidence: 0.6, effort: 10 } },
    { idea: `Pricing sensitivity interviews with 10 ICP prospects`, RICE: { reach: 10, impact: 2, confidence: 0.7, effort: 8 } },
    { idea: `Competitor teardown: top 3`, RICE: { reach: 0, impact: 1, confidence: 0.8, effort: 6 } },
  ];
  if (findings.length > 0) return base;
  return base;
}

export const summarizeBrief = { summarize, generateIdeas };

