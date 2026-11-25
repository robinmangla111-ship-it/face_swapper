// ─── Types ───────────────────────────────────────────────────────────────────

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  total_frames: number;
  duration: number;
  codec: string;
  has_audio: boolean;
  file_size_mb: number;
}

export interface UploadVideoResponse {
  video_id: string;
  filename: string;
  info: VideoInfo;
}

export interface Character {
  id: string;
  label: string;
  face_count: number;
  thumbnail: string; // base64 JPEG
  best_score: number;
}

export interface DetectFacesResponse {
  video_id: string;
  characters: Character[];
  total_characters: number;
}

export interface UploadSourceFaceResponse {
  face_id: string;
  filename: string;
  thumbnail: string; // base64 JPEG
  confidence: number;
  face_count: number;
  swap_mode?: string;
}

export interface LibraryCharacter {
  id: string;
  name: string;
  category: string;
  swap_mode: string;
  filename: string;
  image_url: string;
  is_custom: boolean;
}

export interface SwapRequest {
  video_id: string;
  source_face_id: string;
  character_id: string;
  enhance: boolean;
  num_workers: number | null;
  swap_mode?: string;
}

export interface SwapResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number; // 0-1
  current_step: string;
  frames_processed: number;
  total_frames: number;
  elapsed_seconds: number;
  eta_seconds: number;
  error: string | null;
  output_path: string | null;
}

// ─── API Client ──────────────────────────────────────────────────────────────

const BACKEND_URL =
  typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_API_URL || `http://${window.location.hostname}:8000`)
    : "http://localhost:8000";

const API_BASE = `${BACKEND_URL}/api`;

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let msg = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      msg = body.detail || body.message || msg;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(msg, res.status);
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ───────────────────────────────────────────────────────────────

export async function uploadVideo(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadVideoResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload/video`);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let msg = `Upload failed with status ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          msg = body.detail || body.message || msg;
        } catch {
          // ignore
        }
        reject(new ApiError(msg, xhr.status));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));

    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

export async function detectFaces(videoId: string): Promise<DetectFacesResponse> {
  return request<DetectFacesResponse>(`${API_BASE}/faces/${videoId}`);
}

export async function uploadSourceFace(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadSourceFaceResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload/source-face`);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let msg = `Upload failed with status ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          msg = body.detail || body.message || msg;
        } catch {
          // ignore
        }
        reject(new ApiError(msg, xhr.status));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload aborted")));

    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

export async function startSwap(body: SwapRequest): Promise<SwapResponse> {
  return request<SwapResponse>(`${API_BASE}/swap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`${API_BASE}/jobs/${jobId}`);
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}

export function createProgressWebSocket(
  jobId: string,
  onMessage: (data: JobStatus) => void,
  onError?: (err: Event) => void,
  onClose?: () => void
): WebSocket {
  const wsUrl = BACKEND_URL.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsUrl}/ws/progress/${jobId}`);

  ws.addEventListener("message", (e) => {
    try {
      const data = JSON.parse(e.data) as JobStatus;
      onMessage(data);
    } catch {
      // ignore parse errors
    }
  });

  if (onError) ws.addEventListener("error", onError);
  if (onClose) ws.addEventListener("close", onClose);

  return ws;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await request<unknown>(`${API_BASE}/health`);
    return true;
  } catch {
    return false;
  }
}

// ─── Logs Management ──────────────────────────────────────────────────────────

export interface LogFileInfo {
  filename: string;
  path: string;
  size_kb: number;
  created_at: string;
  modified_at: string;
  is_current: boolean;
}

export interface LogContentResponse {
  filename: string;
  content: string;
  size_kb?: number;
  error_found: boolean;
}

export async function getLogsList(): Promise<LogFileInfo[]> {
  const data = await request<{ logs: LogFileInfo[] }>(`${API_BASE}/logs`);
  return data.logs;
}

export async function getLatestLog(): Promise<LogContentResponse> {
  return request<LogContentResponse>(`${API_BASE}/logs/latest`);
}

export async function getLogContent(filename: string): Promise<LogContentResponse> {
  return request<LogContentResponse>(`${API_BASE}/logs/${filename}`);
}

export async function clearAllLogs(): Promise<{ deleted: number; message: string }> {
  return request<{ deleted: number; message: string }>(`${API_BASE}/logs`, {
    method: "DELETE",
  });
}

// ─── Character Library ───────────────────────────────────────────────────────

export function getLibraryImageUrl(imageUrl: string): string {
  if (imageUrl.startsWith("http")) return imageUrl;
  return `${BACKEND_URL}${imageUrl}`;
}

export async function getLibraryCharacters(category?: string): Promise<LibraryCharacter[]> {
  const query = category && category !== "All" ? `?category=${encodeURIComponent(category)}` : "";
  const data = await request<{ characters: LibraryCharacter[] }>(`${API_BASE}/library${query}`);
  return data.characters;
}

export async function selectLibraryCharacter(charId: string): Promise<UploadSourceFaceResponse> {
  return request<UploadSourceFaceResponse>(`${API_BASE}/library/${encodeURIComponent(charId)}/select`, {
    method: "POST",
  });
}

export async function addLibraryCharacter(
  file: File,
  name: string,
  category: string,
  swapMode: string = "auto"
): Promise<LibraryCharacter> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", name);
  fd.append("category", category);
  fd.append("swap_mode", swapMode);
  const data = await request<{ character: LibraryCharacter }>(`${API_BASE}/library`, {
    method: "POST",
    body: fd,
  });
  return data.character;
}

export async function deleteLibraryCharacter(charId: string): Promise<void> {
  await request<{ deleted: boolean }>(`${API_BASE}/library/${encodeURIComponent(charId)}`, {
    method: "DELETE",
  });
}
