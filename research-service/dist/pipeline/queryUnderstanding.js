"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.expandQueries = void 0;
const DEFAULT_FACETS = [
    'Market size & growth',
    'Buyer roles & cycles',
    'Channels & CAC benchmarks',
    'Top competitors & positioning',
    'Pricing models',
    'Regulations & compliance',
];
function extractEntities(query) {
    const tokens = query.split(/[^a-zA-Z0-9+]+/).filter(Boolean);
    return Array.from(new Set(tokens)).slice(0, 10);
}
function expandKeyQuestions(_query) {
    return DEFAULT_FACETS;
}
function expandSubQueries(query, geo, scope) {
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
exports.expandQueries = {
    extractEntities,
    expandKeyQuestions,
    expandSubQueries,
};
