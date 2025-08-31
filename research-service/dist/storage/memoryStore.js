"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.memoryStore = void 0;
const briefs = new Map();
function saveBrief(brief) {
    briefs.set(brief.briefId, brief);
}
function getBrief(id) {
    return briefs.get(id);
}
exports.memoryStore = { saveBrief, getBrief };
