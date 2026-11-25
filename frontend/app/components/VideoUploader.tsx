"use client";

import React, { useCallback, useRef, useState } from "react";
import { uploadVideo, clearAllLogs, type UploadVideoResponse, type VideoInfo } from "../lib/api";

interface VideoUploaderProps {
  onUploadComplete: (data: UploadVideoResponse) => void;
  onError: (msg: string) => void;
  onOpenLogs?: () => void;
}

const ACCEPTED_TYPES = ["video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/webm"];
const ACCEPTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

export default function VideoUploader({ onUploadComplete, onError, onOpenLogs }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [uploaded, setUploaded] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_TYPES.includes(file.type) && !ACCEPTED_EXTENSIONS.includes(ext)) {
      onError(`Unsupported file type. Please upload: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return false;
    }
    return true;
  };

  const handleUpload = useCallback(
    async (file: File) => {
      if (!validateFile(file)) return;
      setFileName(file.name);
      setUploading(true);
      setProgress(0);

      // Create local preview URL
      const url = URL.createObjectURL(file);
      setVideoUrl(url);

      try {
        const data = await uploadVideo(file, (pct) => setProgress(pct));
        setVideoInfo(data.info);
        setUploaded(true);
        setUploading(false);
        onUploadComplete(data);
      } catch (err: unknown) {
        setUploading(false);
        setVideoUrl(null);
        const message = err instanceof Error ? err.message : "Upload failed";
        onError(message);
      }
    },
    [onUploadComplete, onError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  // After upload is done, show video info
  if (uploaded && videoInfo) {
    return (
      <div className="animate-fade-in">
        <div className="card p-6 space-y-5">
          {/* Video preview */}
          <div className="relative rounded-xl overflow-hidden bg-black aspect-video max-w-lg mx-auto">
            {videoUrl && (
              <video
                src={videoUrl}
                className="w-full h-full object-contain"
                controls
                muted
                playsInline
              />
            )}
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetadataCard label="Resolution" value={`${videoInfo.width}×${videoInfo.height}`} icon="📐" />
            <MetadataCard label="Duration" value={formatDuration(videoInfo.duration)} icon="⏱️" />
            <MetadataCard label="FPS" value={videoInfo.fps.toFixed(1)} icon="🎞️" />
            <MetadataCard label="Size" value={formatSize(videoInfo.file_size_mb)} icon="💾" />
          </div>

          <div className="flex items-center gap-3 text-sm text-zinc-400">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-zinc-800 rounded-lg text-xs font-medium">
              {videoInfo.codec.toUpperCase()}
            </span>
            {videoInfo.has_audio && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-zinc-800 rounded-lg text-xs font-medium">
                🔊 Audio
              </span>
            )}
            <span className="text-zinc-500">{fileName}</span>
          </div>
        </div>
      </div>
    );
  }

  // Upload UI
  return (
    <div className="animate-fade-in">
      <div
        className={`drop-zone text-center ${isDragging ? "active" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="hidden"
          onChange={handleFileSelect}
        />

        {uploading ? (
          <div className="space-y-4 py-4">
            <div className="w-16 h-16 mx-auto rounded-full border-4 border-zinc-700 border-t-brand-500 animate-spin" />
            <p className="text-lg font-medium text-zinc-300">Uploading {fileName}...</p>
            {/* Progress bar */}
            <div className="max-w-sm mx-auto">
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-brand-600 to-brand-400 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-zinc-500 mt-2">{progress}%</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4 py-8">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-zinc-800/80 flex items-center justify-center">
              <svg className="w-10 h-10 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-semibold text-zinc-200">
                Drop your video here or{" "}
                <span className="text-brand-400 hover:text-brand-300">browse</span>
              </p>
              <p className="text-sm text-zinc-500 mt-1">
                Supports MP4, AVI, MOV, MKV, WebM
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Quick Log Actions before upload */}
      <div className="flex items-center justify-between text-xs px-2 pt-1 text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block animate-pulse" />
          Ready to upload
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={async (e) => {
              e.stopPropagation();
              try {
                const res = await clearAllLogs();
                alert(`Older logs cleared (${res.deleted} files removed). Ready for clean upload!`);
              } catch {
                alert("Failed to clear logs");
              }
            }}
            className="hover:text-red-400 transition-colors flex items-center gap-1 px-2 py-1 rounded bg-zinc-800/60 border border-zinc-700/50"
            title="Delete all older log files before this upload"
          >
            <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear Old Logs
          </button>
          {onOpenLogs && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenLogs();
              }}
              className="hover:text-blue-400 transition-colors flex items-center gap-1 px-2 py-1 rounded bg-zinc-800/60 border border-zinc-700/50"
            >
              <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              View Logs
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function MetadataCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="bg-zinc-800/60 rounded-xl p-3 text-center">
      <span className="text-lg">{icon}</span>
      <p className="text-sm font-bold text-zinc-100 mt-1">{value}</p>
      <p className="text-xs text-zinc-500">{label}</p>
    </div>
  );
}
