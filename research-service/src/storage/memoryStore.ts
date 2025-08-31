import type { ResearchBrief } from '../types/types';

const briefs = new Map<string, ResearchBrief>();

function saveBrief(brief: ResearchBrief) {
  briefs.set(brief.briefId, brief);
}

function getBrief(id: string): ResearchBrief | undefined {
  return briefs.get(id);
}

export const memoryStore = { saveBrief, getBrief };

