"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.reviewBrief = reviewBrief;
function reviewBrief(input) {
    // Lightweight guard: ensure each finding has at least one evidence URL
    const findings = input.findings.filter(f => f.evidence && f.evidence.length > 0);
    const summary = input.summary || '';
    return { findings, summary };
}
