"use client";

import React, { useEffect, useState } from "react";
import { detectFaces, type Character } from "../lib/api";

interface FaceSelectorProps {
  videoId: string;
  onSelect: (character: Character) => void;
  onError: (msg: string) => void;
  onOpenLogs?: () => void;
  onBackToUpload?: () => void;
}

export default function FaceSelector({
  videoId,
  onSelect,
  onError,
  onOpenLogs,
  onBackToUpload,
}: FaceSelectorProps) {
  const [loading, setLoading] = useState(true);
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedId, setSelectedId] = useState<string | number | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const lastFetchedIdRef = React.useRef<string | null>(null);

  const runDetection = () => {
    setLoading(true);
    setFetchError(null);

    detectFaces(videoId)
      .then((data) => {
        setCharacters(data.characters || []);
        setLoading(false);
        if (data.characters && data.characters.length > 0) {
          // Pre-select the first character by default
          setSelectedId(data.characters[0].id);
          onSelect(data.characters[0]);
        }
      })
      .catch((err) => {
        setLoading(false);
        const msg = err instanceof Error ? err.message : "Failed to detect faces";
        setFetchError(msg);
        onError(msg);
      });
  };

  useEffect(() => {
    if (lastFetchedIdRef.current === videoId) return;
    lastFetchedIdRef.current = videoId;
    runDetection();
  }, [videoId]);

  const handleSelect = (char: Character) => {
    setSelectedId(char.id);
    onSelect(char);
  };

  if (loading) {
    return (
      <div className="animate-fade-in text-center py-16 space-y-5">
        <div className="relative w-24 h-24 mx-auto">
          <div className="absolute inset-0 rounded-full border-4 border-zinc-800" />
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-500 animate-spin" />
          <div className="absolute inset-3 rounded-full border-4 border-transparent border-b-cyan-400 animate-spin-slow" />
          <div className="absolute inset-0 flex items-center justify-center text-xl">
            🔍
          </div>
        </div>
        <div className="space-y-1">
          <p className="text-xl font-bold text-zinc-100">Scanning &amp; Identifying Characters...</p>
          <p className="text-sm text-zinc-400">
            Analyzing video frames with InsightFace buffalo_l engine
          </p>
        </div>
      </div>
    );
  }

  // Error State
  if (fetchError) {
    return (
      <div className="animate-fade-in text-center py-12 px-6 max-w-lg mx-auto bg-red-950/20 border border-red-800/40 rounded-2xl space-y-4">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-red-900/30 flex items-center justify-center text-2xl text-red-400">
          ⚠️
        </div>
        <div className="space-y-1">
          <h3 className="text-lg font-bold text-red-200">Face Detection Failed</h3>
          <p className="text-xs text-red-300 font-mono bg-red-950/60 p-2.5 rounded-lg border border-red-900/50 break-words">
            {fetchError}
          </p>
        </div>
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={runDetection}
            className="px-4 py-2 text-xs font-semibold rounded-xl bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-md flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Retry Detection
          </button>
          {onOpenLogs && (
            <button
              onClick={onOpenLogs}
              className="px-4 py-2 text-xs font-medium rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition-all flex items-center gap-1.5"
            >
              <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              View Error Logs
            </button>
          )}
          {onBackToUpload && (
            <button
              onClick={onBackToUpload}
              className="px-3 py-2 text-xs text-zinc-400 hover:text-white transition-colors"
            >
              Back to Upload
            </button>
          )}
        </div>
      </div>
    );
  }

  // No faces detected in video
  if (characters.length === 0) {
    return (
      <div className="animate-fade-in text-center py-14 space-y-4 max-w-md mx-auto">
        <div className="w-20 h-20 mx-auto rounded-3xl bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center text-3xl">
          👤
        </div>
        <div className="space-y-1">
          <p className="text-xl font-bold text-zinc-200">No Faces Detected in Video</p>
          <p className="text-sm text-zinc-500">
            The video may have fast cuts, small faces, or poor lighting. Try another clip or re-scan.
          </p>
        </div>
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={runDetection}
            className="px-4 py-2 text-xs font-semibold rounded-xl bg-blue-600 hover:bg-blue-500 text-white transition-all shadow-md"
          >
            Re-scan Video
          </button>
          {onBackToUpload && (
            <button
              onClick={onBackToUpload}
              className="px-4 py-2 text-xs font-medium rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition-all"
            >
              Upload Different Video
            </button>
          )}
          {onOpenLogs && (
            <button
              onClick={onOpenLogs}
              className="px-3 py-2 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              Check Logs
            </button>
          )}
        </div>
      </div>
    );
  }

  // Characters Found
  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">
            Detected Characters ({characters.length})
          </h3>
          <p className="text-xs text-zinc-400">
            Click on the person whose face you want to replace
          </p>
        </div>
        <span className="text-xs px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
          Ready for Swap
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {characters.map((char) => {
          const isSelected = String(selectedId) === String(char.id);
          return (
            <button
              key={char.id}
              onClick={() => handleSelect(char)}
              className={`
                group relative p-3 text-left rounded-2xl transition-all duration-300 border flex flex-col justify-between
                ${
                  isSelected
                    ? "bg-blue-950/40 border-blue-500 ring-2 ring-blue-500/40 shadow-xl shadow-blue-500/20 scale-[1.02]"
                    : "bg-zinc-900/80 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/60"
                }
              `}
            >
              {/* Checkmark badge */}
              {isSelected && (
                <div className="absolute -top-2 -right-2 w-7 h-7 bg-blue-500 rounded-full flex items-center justify-center shadow-lg shadow-blue-500/50 z-10 animate-scale-in">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={3}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              )}

              {/* Face Crop Thumbnail */}
              <div className="aspect-square rounded-xl overflow-hidden bg-zinc-950 mb-3 border border-zinc-800/80 shadow-inner relative">
                {char.thumbnail ? (
                  <img
                    src={`data:image/jpeg;base64,${char.thumbnail}`}
                    alt={char.label}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-600 text-3xl">
                    👤
                  </div>
                )}
                {char.best_score && (
                  <span className="absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-black/70 text-emerald-400 backdrop-blur-xs">
                    {(char.best_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Info */}
              <div className="space-y-0.5">
                <p className="text-sm font-bold text-zinc-100 truncate">{char.label}</p>
                <p className="text-xs text-zinc-400">
                  {char.face_count} detection{char.face_count !== 1 ? "s" : ""}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
