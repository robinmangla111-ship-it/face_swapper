"use client";

import React, { useEffect, useRef, useState } from "react";
import { createProgressWebSocket, getJobStatus, type JobStatus } from "../lib/api";

interface ProcessingViewProps {
  jobId: string;
  onComplete: (job: JobStatus) => void;
  onError: (msg: string) => void;
  onRetry?: () => void;
  onOpenLogs?: () => void;
}

function formatTime(seconds: number): string {
  if (!seconds || seconds < 0) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ProcessingView({
  jobId,
  onComplete,
  onError,
  onRetry,
  onOpenLogs,
}: ProcessingViewProps) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const completedRef = useRef(false);

  const handleStatus = (data: JobStatus) => {
    setStatus(data);

    if (data.status === "completed" && !completedRef.current) {
      completedRef.current = true;
      onComplete(data);
    } else if (data.status === "failed" && !completedRef.current) {
      completedRef.current = true;
      onError(data.error || "Processing failed");
    }
  };

  useEffect(() => {
    completedRef.current = false;

    // Try WebSocket first
    try {
      const ws = createProgressWebSocket(
        jobId,
        handleStatus,
        () => {
          // On WebSocket error, fall back to polling
          startPolling();
        },
        () => {
          // On close, start polling if not completed
          if (!completedRef.current) startPolling();
        }
      );
      wsRef.current = ws;
    } catch {
      startPolling();
    }

    function startPolling() {
      if (pollingRef.current || completedRef.current) return;
      pollingRef.current = setInterval(async () => {
        try {
          const data = await getJobStatus(jobId);
          handleStatus(data);
          if (data.status === "completed" || data.status === "failed") {
            if (pollingRef.current) clearInterval(pollingRef.current);
          }
        } catch {
          // ignore polling errors
        }
      }, 2000);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const rawProgress = status?.progress;
  const progressPct =
    typeof rawProgress === "number" && !isNaN(rawProgress)
      ? Math.min(100, Math.max(0, Math.round(rawProgress * 100)))
      : 0;
  const circumference = 2 * Math.PI * 54; // radius=54
  const strokeOffset = circumference - (progressPct / 100) * circumference;

  const isFailed = status?.status === "failed";

  if (isFailed) {
    return (
      <div className="animate-fade-in text-center py-8 space-y-6 max-w-md mx-auto">
        <div className="w-16 h-16 rounded-2xl bg-red-950/80 border border-red-800/60 flex items-center justify-center mx-auto text-red-400 shadow-xl shadow-red-950/50">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>

        <div className="space-y-2">
          <h3 className="text-xl font-bold text-zinc-100">Face Swap Failed</h3>
          <p className="text-sm text-red-300/90 font-mono bg-red-950/40 p-3 rounded-xl border border-red-900/40 break-words text-left max-h-36 overflow-y-auto">
            {status?.error || "Processing failed during video rendering."}
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 justify-center pt-2">
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white font-medium shadow-lg shadow-brand-500/25 transition-all text-sm flex items-center justify-center gap-2"
            >
              🔄 Try Again
            </button>
          )}
          {onOpenLogs && (
            <button
              onClick={onOpenLogs}
              className="px-5 py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-medium border border-zinc-700 transition-all text-sm flex items-center justify-center gap-2"
            >
              📋 View Error Logs
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in text-center py-8 space-y-8">
      {/* Circular progress */}
      <div className="relative w-40 h-40 mx-auto">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          {/* Background track */}
          <circle
            cx="60" cy="60" r="54"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-zinc-800"
          />
          {/* Progress arc */}
          <circle
            cx="60" cy="60" r="54"
            fill="none"
            stroke="url(#progressGradient)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeOffset}
            className="transition-all duration-500 ease-out"
          />
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-gradient">{progressPct}%</span>
        </div>

        {/* Animated ring glow */}
        <div className="absolute inset-0 rounded-full animate-pulse-slow opacity-30"
          style={{
            background: `radial-gradient(circle, transparent 50%, rgba(59,130,246,0.1) 100%)`,
          }}
        />
      </div>

      {/* Status info */}
      <div className="space-y-3">
        <p className="text-lg font-medium text-zinc-200">
          {status?.current_step || "Starting..."}
        </p>

        {status && status.frames_processed > 0 && (
          <p className="text-sm text-zinc-400">
            {status.frames_processed.toLocaleString()} / {status.total_frames.toLocaleString()} frames
          </p>
        )}

        {/* Progress bar (linear, complementing the circle) */}
        <div className="max-w-sm mx-auto">
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-600 to-cyan-500 rounded-full transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {/* Time info */}
        <div className="flex justify-center gap-8 text-sm">
          <div>
            <span className="text-zinc-500">Elapsed</span>
            <p className="text-zinc-300 font-mono font-medium">
              {formatTime(status?.elapsed_seconds ?? 0)}
            </p>
          </div>
          <div>
            <span className="text-zinc-500">ETA</span>
            <p className="text-zinc-300 font-mono font-medium">
              {formatTime(status?.eta_seconds ?? 0)}
            </p>
          </div>
        </div>
      </div>

      {/* Animated dots */}
      <div className="flex justify-center gap-1.5">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-brand-500 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}
