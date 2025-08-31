import type { Finding } from '../types/types';

export function reviewBrief(input: { findings: Finding[]; summary?: string }) {
  // Lightweight guard: ensure each finding has at least one evidence URL
  const findings = input.findings.filter(f => f.evidence && f.evidence.length > 0);
  const summary = input.summary || '';
  return { findings, summary };
}

