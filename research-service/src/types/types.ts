import { z } from 'zod';

export type Evidence = {
  quote: string;
  url: string;
  published?: string;
};

export type Finding = {
  title: string;
  summary: string;
  evidence: Evidence[];
  confidence: number; // 0..1
  recency?: string;
};

export type RICE = {
  reach: number;
  impact: number; // 0..3
  confidence: number; // 0..1
  effort: number; // points
};

export type Idea = {
  idea: string;
  RICE: RICE;
};

export type AttachmentRef = {
  type: 'pdf' | 'screenshot' | 'other';
  path: string;
};

export type ResearchBrief = {
  briefId: string;
  query: string;
  date: string; // ISO
  entities: string[];
  keyQuestions: string[];
  findings: Finding[];
  ideas: Idea[];
  attachments: AttachmentRef[];
  summary?: string;
};

export type PlanJSON = {
  okrs: Array<{
    objective: string;
    keyResults: Array<{ metric: string; target: number }>;
  }>;
  initiatives: Array<{
    id: string;
    title: string;
    tasks: Array<{ id: string; title: string; assignee: string }>;
  }>;
};

export const NotesSchema = z.object({
  notesId: z.string().uuid().optional(),
  description: z.string().optional(),
  subject: z.string().optional(),
  leadId: z.string().uuid().optional(),
  leadName: z.string().optional(),
  createdById: z.string().uuid().optional(),
  createdByName: z.string().optional(),
});

export type Notes = z.infer<typeof NotesSchema>;

export type NotesAttachment = { url: string; name?: string };

export const AttachmentSchema = z.object({
  attachmentId: z.string().uuid().optional(),
  title: z.string().optional(),
  description: z.string().optional(),
  attachmentUrl: z.array(z.object({ url: z.string().url(), name: z.string().optional() })).optional(),
  leadId: z.string().uuid().optional(),
  createdById: z.string().uuid().optional(),
  createdByName: z.string().optional(),
  taskId: z.string().uuid().optional(),
});

export type CRMTask = {
  id?: string;
  name: string;
  taskStatus?: 'NEW' | 'NOT_STARTED' | 'IN_PROGRESS' | 'COMPLETED' | 'WAITING_FOR_INPUT' | 'CANCELLED';
  businessId?: string;
  assignedTo?: string;
  assignedName?: string;
  description?: string;
  dueDate?: string;
  priority?: 'NEW' | 'HIGH' | 'MEDIUM' | 'LOW';
  createdById?: string;
  createdByName?: string;
};

