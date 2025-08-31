"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.crmClientFactory = crmClientFactory;
const got_1 = __importDefault(require("got"));
const zod_1 = require("zod");
const types_1 = require("../types/types");
const NotesResponseSchema = zod_1.z.object({ data: zod_1.z.any().optional(), notesId: zod_1.z.string().uuid().optional() }).passthrough();
function crmClientFactory({ baseUrl, bearerToken }) {
    const client = got_1.default.extend({
        prefixUrl: baseUrl.replace(/\/$/, ''),
        headers: bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {},
        timeout: { request: 10000 },
    });
    async function createNote(note) {
        const payload = types_1.NotesSchema.partial().parse(note);
        const resp = await client.post('notes/create', { json: payload }).json();
        const parsed = NotesResponseSchema.safeParse(resp);
        return { notesId: parsed.success ? parsed.data.notesId : undefined, raw: resp };
    }
    async function updateNote(note) {
        const payload = types_1.NotesSchema.parse(note);
        const resp = await client.put('notes/update', { json: payload }).json();
        return resp;
    }
    async function createTaskAttachment(taskId, attachment) {
        const payload = types_1.AttachmentSchema.parse(attachment);
        const url = `task/attachment/create/${taskId}`;
        const resp = await client.post(url, { json: payload }).json();
        return resp;
    }
    async function createTask(task) {
        const resp = await client.post('task/create', { json: task }).json();
        return resp;
    }
    async function changeTaskStatus(taskId, taskStatus) {
        const resp = await client.put(`task/change/status`, { searchParams: { taskId, taskStatus } }).json();
        return resp;
    }
    return { createNote, updateNote, createTaskAttachment, createTask, changeTaskStatus };
}
