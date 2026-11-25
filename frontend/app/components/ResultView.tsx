"use client";

import React from "react";
import { getDownloadUrl } from "../lib/api";

interface ResultViewProps {
  jobId: string;
  onReset: () => void;
}

export default function ResultView({ jobId, onReset }: ResultViewProps) {
  const downloadUrl = getDownloadUrl(jobId);

  return (
    <div className="animate-fade-in space-y-8">
      {/* Success banner */}
      <div className="text-center space-y-2">
        <div className="w-16 h-16 mx-auto rounded-full bg-green-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-zinc-100">Face Swap Complete!</h2>
        <p className="text-zinc-400">Your video is ready to download</p>
      </div>

      {/* Video player */}
      <div className="card p-4 max-w-2xl mx-auto">
        <div className="rounded-xl overflow-hidden bg-black aspect-video">
          <video
            src={downloadUrl}
            className="w-full h-full object-contain"
            controls
            autoPlay
            muted
            playsInline
          />
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <a
          href={downloadUrl}
          download
          className="btn-primary inline-flex items-center gap-2 text-lg px-8 py-4"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          Download Video
        </a>

        <button
          onClick={onReset}
          className="btn-secondary inline-flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
          </svg>
          Start New Swap
        </button>
      </div>
    </div>
  );
}
