"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.summarizeBrief = void 0;
async function summarize(findings, query) {
    const bullets = findings.slice(0, 5).map(f => `- ${f.title}: ${f.summary}`);
    return [`Brief for: ${query}`, ...bullets].join('\n');
}
function generateIdeas(findings, query) {
    const base = [
        { idea: `Validate top 2 channels inferred from findings for: ${query}`, RICE: { reach: 1000, impact: 2, confidence: 0.6, effort: 10 } },
        { idea: `Pricing sensitivity interviews with 10 ICP prospects`, RICE: { reach: 10, impact: 2, confidence: 0.7, effort: 8 } },
        { idea: `Competitor teardown: top 3`, RICE: { reach: 0, impact: 1, confidence: 0.8, effort: 6 } },
    ];
    if (findings.length > 0)
        return base;
    return base;
}
exports.summarizeBrief = { summarize, generateIdeas };
