"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.fetchAndExtract = fetchAndExtract;
const got_1 = __importDefault(require("got"));
const jsdom_1 = require("jsdom");
const readability_1 = require("@mozilla/readability");
async function fetchOne(hit, logger) {
    try {
        const resp = await (0, got_1.default)(hit.url, { timeout: { request: 8000 } });
        const dom = new jsdom_1.JSDOM(resp.body, { url: hit.url });
        const reader = new readability_1.Readability(dom.window.document);
        const article = reader.parse();
        const contentText = article?.textContent?.trim() || '';
        if (!contentText)
            return null;
        return {
            url: hit.url,
            title: article?.title || hit.title,
            contentText,
            published: hit.published,
        };
    }
    catch (err) {
        logger.debug({ url: hit.url, err: String(err) }, 'fetch failed');
        return null;
    }
}
async function fetchAndExtract(hits, logger) {
    const tasks = hits.slice(0, 20).map(h => fetchOne(h, logger));
    const results = await Promise.all(tasks);
    return results.filter((r) => Boolean(r));
}
