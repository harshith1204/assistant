"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runResearch = runResearch;
const uuid_1 = require("uuid");
const queryUnderstanding_1 = require("../pipeline/queryUnderstanding");
const search_1 = require("../pipeline/search");
const fetcher_1 = require("../pipeline/fetcher");
const normalizer_1 = require("../pipeline/normalizer");
const credibility_1 = require("../pipeline/credibility");
const extraction_1 = require("../pipeline/extraction");
const summarize_1 = require("../pipeline/summarize");
const review_1 = require("../pipeline/review");
async function runResearch(input, logger) {
    const now = new Date();
    const briefId = (0, uuid_1.v4)();
    const entities = queryUnderstanding_1.expandQueries.extractEntities(input.query);
    const keyQuestions = queryUnderstanding_1.expandQueries.expandKeyQuestions(input.query);
    const subQueries = queryUnderstanding_1.expandQueries.expandSubQueries(input.query, input.geo, input.scope);
    const searchResults = await (0, search_1.searchWeb)(subQueries, input.freshness, logger);
    const fetched = await (0, fetcher_1.fetchAndExtract)(searchResults, logger);
    const normalized = (0, normalizer_1.normalizeAndCluster)(fetched);
    const scored = (0, credibility_1.scoreCredibility)(normalized);
    const topDocs = scored.slice(0, 20);
    const findings = (0, extraction_1.extractFacts)(topDocs);
    const ideas = summarize_1.summarizeBrief.generateIdeas(findings, input.query);
    const summary = await summarize_1.summarizeBrief.summarize(findings, input.query);
    const reviewed = (0, review_1.reviewBrief)({ findings, summary });
    const brief = {
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
