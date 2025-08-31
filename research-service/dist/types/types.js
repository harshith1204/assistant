"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AttachmentSchema = exports.NotesSchema = void 0;
const zod_1 = require("zod");
exports.NotesSchema = zod_1.z.object({
    notesId: zod_1.z.string().uuid().optional(),
    description: zod_1.z.string().optional(),
    subject: zod_1.z.string().optional(),
    leadId: zod_1.z.string().uuid().optional(),
    leadName: zod_1.z.string().optional(),
    createdById: zod_1.z.string().uuid().optional(),
    createdByName: zod_1.z.string().optional(),
});
exports.AttachmentSchema = zod_1.z.object({
    attachmentId: zod_1.z.string().uuid().optional(),
    title: zod_1.z.string().optional(),
    description: zod_1.z.string().optional(),
    attachmentUrl: zod_1.z.array(zod_1.z.object({ url: zod_1.z.string().url(), name: zod_1.z.string().optional() })).optional(),
    leadId: zod_1.z.string().uuid().optional(),
    createdById: zod_1.z.string().uuid().optional(),
    createdByName: zod_1.z.string().optional(),
    taskId: zod_1.z.string().uuid().optional(),
});
