"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.searchWeb = searchWeb;
async function searchWeb(queries, _freshness, logger) {
    // Placeholder: integrate with Bing/Google/News via env keys.
    // For now, return unique URLs guessed from queries to keep flow working in dev.
    const hits = [];
    const seen = new Set();
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
