import got from 'got';
import { z } from 'zod';
import { AttachmentSchema, NotesSchema } from '../types/types';

type FactoryInput = { baseUrl: string; bearerToken?: string };

const NotesResponseSchema = z.object({ data: z.any().optional(), notesId: z.string().uuid().optional() }).passthrough();

export function crmClientFactory({ baseUrl, bearerToken }: FactoryInput) {
  const client = got.extend({
    prefixUrl: baseUrl.replace(/\/$/, ''),
    headers: bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {},
    timeout: { request: 10000 },
  });

  async function createNote(note: z.infer<typeof NotesSchema> & { subject?: string; description?: string }) {
    const payload = NotesSchema.partial().parse(note);
    const resp = await client.post('notes/create', { json: payload }).json<any>();
    const parsed = NotesResponseSchema.safeParse(resp);
    return { notesId: parsed.success ? parsed.data.notesId : undefined, raw: resp };
  }

  async function updateNote(note: z.infer<typeof NotesSchema>) {
    const payload = NotesSchema.parse(note);
    const resp = await client.put('notes/update', { json: payload }).json<any>();
    return resp;
  }

  async function createTaskAttachment(taskId: string, attachment: z.infer<typeof AttachmentSchema>) {
    const payload = AttachmentSchema.parse(attachment);
    const url = `task/attachment/create/${taskId}`;
    const resp = await client.post(url, { json: payload }).json<any>();
    return resp;
  }

  async function createTask(task: any) {
    const resp = await client.post('task/create', { json: task }).json<any>();
    return resp;
  }

  async function changeTaskStatus(taskId: string, taskStatus: string) {
    const resp = await client.put(`task/change/status`, { searchParams: { taskId, taskStatus } }).json<any>();
    return resp;
  }

  return { createNote, updateNote, createTaskAttachment, createTask, changeTaskStatus };
}

