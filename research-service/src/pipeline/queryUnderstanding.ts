const DEFAULT_FACETS = [
  'Market size & growth',
  'Buyer roles & cycles',
  'Channels & CAC benchmarks',
  'Top competitors & positioning',
  'Pricing models',
  'Regulations & compliance',
];

function extractEntities(query: string): string[] {
  const tokens = query.split(/[^a-zA-Z0-9+]+/).filter(Boolean);
  return Array.from(new Set(tokens)).slice(0, 10);
}

function expandKeyQuestions(_query: string): string[] {
  return DEFAULT_FACETS;
}

function expandSubQueries(query: string, geo?: string[], scope?: string[]): string[] {
  const base = [query];
  const facets = DEFAULT_FACETS.map(f => `${query} ${f}`);
  const geoVariants = (geo || []).flatMap(g => [
    `${query} ${g} market size`,
    `${query} ${g} competitors`,
    `${query} ${g} pricing benchmarks`,
  ]);
  const scopeVariants = (scope || []).map(s => `${query} ${s}`);
  return Array.from(new Set([...base, ...facets, ...geoVariants, ...scopeVariants])).slice(0, 25);
}

export const expandQueries = {
  extractEntities,
  expandKeyQuestions,
  expandSubQueries,
};

