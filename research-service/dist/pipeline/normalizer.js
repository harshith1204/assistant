"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeAndCluster = normalizeAndCluster;
function simpleHash(text) {
    let h = 0;
    for (let i = 0; i < text.length; i++) {
        h = (h * 31 + text.charCodeAt(i)) | 0;
    }
    return String(h);
}
function normalizeAndCluster(docs) {
    const dedup = new Map();
    for (const d of docs) {
        const hash = simpleHash(d.contentText.slice(0, 2000));
        const clusterKey = d.title?.toLowerCase().split(/\s+/).slice(0, 4).join('-') || 'misc';
        const nd = { ...d, hash, clusterKey };
        if (!dedup.has(hash))
            dedup.set(hash, nd);
    }
    return Array.from(dedup.values());
}
