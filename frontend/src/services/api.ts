import type { HistoryItem, Method, ProcessResult } from "../types";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const endpoint = (path: string) => `${API}${path}`;

async function parseError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text) as { detail?: string };
    return parsed.detail || text;
  } catch {
    return text || `Request failed (${response.status})`;
  }
}

export async function health(): Promise<boolean> {
  try { return (await fetch(endpoint("/health"))).ok; } catch { return false; }
}

export async function upload(file: File) {
  const form = new FormData(); form.append("file", file);
  const response = await fetch(endpoint("/api/v1/images/upload"), { method: "POST", body: form });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<{ image_id: string; width: number; height: number; file_size: number }>;
}

export async function process(imageId: string, filename: string, method: Method, budget: number, iterations: number, seed: number) {
  const response = await fetch(endpoint("/api/v1/images/process"), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, filename, method, budget, iterations, seed }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<ProcessResult>;
}

export async function getHistory(): Promise<HistoryItem[]> {
  const response = await fetch(endpoint("/api/v1/images/history"));
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<HistoryItem[]>;
}

export function assetUrl(path: string, cacheBust?: number) {
  const url = path.startsWith("http") ? path : endpoint(path);
  if (!cacheBust) return url;
  return `${url}${url.includes("?") ? "&" : "?"}t=${cacheBust}`;
}
export { API };

export async function protectImage(imageId: string, filename: string, budget: number, iterations: number, seed: number) {
  const response = await fetch(endpoint("/api/v1/protection/optimize"), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, filename, budget, iterations, seed, eot_samples: 3 }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json() as Promise<ProcessResult & {
    configured_models: string[];
    objective_before: number;
    objective_after: number;
    max_channel_delta: number;
  }>;
}
