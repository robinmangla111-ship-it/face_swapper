"use client";

import React, { useEffect, useState } from "react";
import {
  getLogsList,
  getLatestLog,
  getLogContent,
  clearAllLogs,
  type LogFileInfo,
  type LogContentResponse,
} from "../lib/api";

interface LogsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLogsCleared?: () => void;
}

export default function LogsModal({ isOpen, onClose, onLogsCleared }: LogsModalProps) {
  const [logs, setLogs] = useState<LogFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [logData, setLogData] = useState<LogContentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const loadLogsList = async () => {
    try {
      const list = await getLogsList();
      setLogs(list);
      if (list.length > 0 && !selectedFile) {
        setSelectedFile(list[0].filename);
      }
    } catch (e) {
      console.error("Failed to load logs list", e);
    }
  };

  const loadContent = async (filename?: string) => {
    setLoading(true);
    try {
      const data = filename ? await getLogContent(filename) : await getLatestLog();
      setLogData(data);
    } catch (e) {
      setLogData({
        filename: filename || "none",
        content: `Error loading log: ${e}`,
        error_found: true,
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadLogsList();
      loadContent(selectedFile || undefined);
    }
  }, [isOpen, selectedFile]);

  const handleClearAll = async () => {
    if (!confirm("Are you sure you want to delete all older log files?")) return;
    try {
      const res = await clearAllLogs();
      setActionMessage(res.message);
      setLogs([]);
      setLogData(null);
      setSelectedFile(null);
      if (onLogsCleared) onLogsCleared();
      setTimeout(() => setActionMessage(null), 3000);
    } catch (e) {
      setActionMessage(`Failed to clear logs: ${e}`);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-zinc-900 border border-zinc-700/80 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/90">
          <div className="flex items-center gap-3">
            <span className="p-2 rounded-xl bg-blue-500/10 text-blue-400">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </span>
            <div>
              <h2 className="text-lg font-bold text-white">System Logs & Diagnostics</h2>
              <p className="text-xs text-zinc-400">Review operations, analyze failures, or clear older session logs</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleClearAll}
              className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all border border-red-500/30 flex items-center gap-1.5"
              title="Delete all older log files before a new upload"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Clear Older Logs
            </button>
            <button
              onClick={() => {
                loadLogsList();
                loadContent(selectedFile || undefined);
              }}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700 transition-all border border-zinc-700 flex items-center gap-1"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {actionMessage && (
          <div className="px-6 py-2 text-xs bg-blue-500/10 border-b border-blue-500/20 text-blue-300 flex items-center justify-between">
            <span>{actionMessage}</span>
          </div>
        )}

        {/* Body */}
        <div className="grid grid-cols-1 md:grid-cols-3 flex-1 overflow-hidden">
          {/* File Selector Sidebar */}
          <div className="border-r border-zinc-800 p-4 space-y-2 overflow-y-auto max-h-[60vh] bg-zinc-950/40">
            <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider px-2">
              Log Files ({logs.length})
            </div>
            {logs.length === 0 ? (
              <div className="text-xs text-zinc-500 p-3 italic">No logs recorded yet.</div>
            ) : (
              logs.map((item) => (
                <button
                  key={item.filename}
                  onClick={() => setSelectedFile(item.filename)}
                  className={`w-full text-left p-2.5 rounded-xl text-xs transition-all flex flex-col gap-0.5 border ${
                    selectedFile === item.filename
                      ? "bg-blue-600/20 text-blue-300 border-blue-500/40 shadow-sm"
                      : "bg-zinc-900/60 text-zinc-400 border-zinc-800 hover:bg-zinc-800 hover:text-zinc-200"
                  }`}
                >
                  <span className="font-mono font-medium truncate">{item.filename}</span>
                  <div className="flex items-center justify-between text-[10px] text-zinc-500">
                    <span>{item.created_at.split(" ")[1]}</span>
                    <span>{item.size_kb} KB</span>
                  </div>
                </button>
              ))
            )}
          </div>

          {/* Log Viewer Content */}
          <div className="md:col-span-2 p-4 flex flex-col overflow-hidden bg-black/60">
            <div className="flex items-center justify-between mb-2 pb-2 border-b border-zinc-800">
              <span className="text-xs font-mono text-zinc-300 truncate">
                {logData?.filename || "No file selected"}
              </span>
              {logData?.error_found && (
                <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                  Errors Detected
                </span>
              )}
            </div>

            <div className="flex-1 overflow-y-auto font-mono text-xs text-zinc-300 bg-zinc-950 p-4 rounded-xl border border-zinc-800/80 leading-relaxed whitespace-pre-wrap select-text">
              {loading ? (
                <div className="text-zinc-500 italic">Loading log content...</div>
              ) : logData?.content ? (
                logData.content.split("\n").map((line, i) => {
                  let color = "text-zinc-300";
                  if (line.includes("[ERROR]") || line.includes("EXCEPTION DETAILS")) {
                    color = "text-red-400 font-bold bg-red-950/30 py-0.5 px-1 rounded";
                  } else if (line.includes("[WARNING]")) {
                    color = "text-yellow-400 font-semibold";
                  } else if (line.includes("===")) {
                    color = "text-blue-400 font-bold";
                  }
                  return (
                    <div key={i} className={color}>
                      {line}
                    </div>
                  );
                })
              ) : (
                <div className="text-zinc-500 italic">Empty log file.</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-zinc-800 bg-zinc-900/90 text-xs text-zinc-500">
          <span>Tip: You can clear older logs prior to any new video upload to keep diagnostics clean.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 font-medium rounded-lg bg-zinc-800 text-zinc-200 hover:bg-zinc-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
