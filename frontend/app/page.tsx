"use client";

import React, { useCallback, useState } from "react";
import StepIndicator from "./components/StepIndicator";
import VideoUploader from "./components/VideoUploader";
import FaceSelector from "./components/FaceSelector";
import SourceFaceUpload from "./components/SourceFaceUpload";
import ProcessingView from "./components/ProcessingView";
import ResultView from "./components/ResultView";
import SettingsPanel, { type SwapSettings } from "./components/SettingsPanel";
import LogsModal from "./components/LogsModal";
import {
  startSwap,
  type UploadVideoResponse,
  type Character,
  type UploadSourceFaceResponse,
  type JobStatus,
} from "./lib/api";

type ToastType = "error" | "success" | "info";

interface Toast {
  id: number;
  type: ToastType;
  message: string;
}

let toastIdCounter = 0;

export default function HomePage() {
  // ─── State Machine ─────────────────────────────────────────────────────
  const [step, setStep] = useState(1);

  // Data from each step
  const [videoData, setVideoData] = useState<UploadVideoResponse | null>(null);
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [sourceFace, setSourceFace] = useState<UploadSourceFaceResponse | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  // Settings
  const [settings, setSettings] = useState<SwapSettings>({
    enhance: true,
    numWorkers: null,
  });

  // Toasts
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Logs modal
  const [showLogs, setShowLogs] = useState(false);

  const addToast = useCallback((type: ToastType, message: string) => {
    const id = ++toastIdCounter;
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 5000);
  }, []);

  const handleError = useCallback(
    (msg: string) => addToast("error", msg),
    [addToast]
  );

  // ─── Step Handlers ─────────────────────────────────────────────────────

  const handleVideoUpload = useCallback(
    (data: UploadVideoResponse) => {
      setVideoData(data);
      addToast("success", "Video uploaded successfully");
      setStep(2);
    },
    [addToast]
  );

  const handleCharacterSelect = useCallback((char: Character) => {
    setSelectedCharacter(char);
  }, []);

  const handleSourceFaceUpload = useCallback(
    (data: UploadSourceFaceResponse) => {
      setSourceFace(data);
      addToast("success", "Source face uploaded");
    },
    [addToast]
  );

  const handleStartSwap = useCallback(async () => {
    if (!videoData || !selectedCharacter || !sourceFace) return;

    try {
      const res = await startSwap({
        video_id: videoData.video_id,
        source_face_id: sourceFace.face_id,
        character_id: selectedCharacter.id,
        enhance: settings.enhance,
        num_workers: settings.numWorkers,
        swap_mode: sourceFace.swap_mode || "auto",
      });
      setJobId(res.job_id);
      setStep(4);
      addToast("info", "Processing started");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to start swap";
      handleError(message);
    }
  }, [videoData, selectedCharacter, sourceFace, settings, addToast, handleError]);

  const handleProcessingComplete = useCallback(
    (_job: JobStatus) => {
      addToast("success", "Face swap completed!");
      setStep(5);
    },
    [addToast]
  );

  const handleReset = useCallback(() => {
    setStep(1);
    setVideoData(null);
    setSelectedCharacter(null);
    setSourceFace(null);
    setJobId(null);
  }, []);

  // ─── Step titles ───────────────────────────────────────────────────────

  const stepTitles: Record<number, { title: string; subtitle: string }> = {
    1: { title: "Upload Your Video", subtitle: "Start by uploading the video you want to modify" },
    2: { title: "Select Character", subtitle: "Choose which person's face to replace" },
    3: { title: "Upload Source Face", subtitle: "Provide the face that will be swapped in" },
    4: { title: "Processing", subtitle: "AI is working its magic — sit tight!" },
    5: { title: "Your Result", subtitle: "Face swap is done — preview and download" },
  };

  const { title, subtitle } = stepTitles[step];

  return (
    <>
      {/* ─── Header ─────────────────────────────────────────────── */}
      <header className="glass sticky top-0 z-50 border-b border-zinc-800/50">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo */}
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-brand-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
              </svg>
            </div>
            <h1 className="text-lg font-bold tracking-tight">
              <span className="text-gradient">Face Swapper</span>
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowLogs(true)}
              className="px-3 py-1.5 text-xs font-medium rounded-xl bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-300 border border-zinc-700/60 transition-all flex items-center gap-1.5 shadow-sm"
              title="Open System Diagnostics & Logs"
            >
              <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Diagnostics &amp; Logs
            </button>
            <span className="text-xs text-zinc-600 hidden sm:block">AI-Powered Face Replacement</span>
          </div>
        </div>
      </header>

      {/* ─── Main Content ───────────────────────────────────────── */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-8 space-y-8">
        {/* Step indicator */}
        <StepIndicator currentStep={step} />

        {/* Step heading */}
        <div className="text-center space-y-1 pt-2">
          <h2 className="text-2xl font-bold text-zinc-100">{title}</h2>
          <p className="text-sm text-zinc-500">{subtitle}</p>
        </div>

        {/* Step content */}
        <div className="min-h-[300px]">
          {step === 1 && (
            <VideoUploader
              onUploadComplete={handleVideoUpload}
              onError={handleError}
              onOpenLogs={() => setShowLogs(true)}
            />
          )}

          {step === 2 && videoData && (
            <div className="space-y-6">
              <FaceSelector
                videoId={videoData.video_id}
                onSelect={handleCharacterSelect}
                onError={handleError}
                onOpenLogs={() => setShowLogs(true)}
                onBackToUpload={() => setStep(1)}
              />
              {selectedCharacter && (
                <div className="flex justify-center animate-fade-in">
                  <button onClick={() => setStep(3)} className="btn-primary">
                    Continue with {selectedCharacter.label}
                  </button>
                </div>
              )}
            </div>
          )}

          {step === 3 && selectedCharacter && (
            <div className="space-y-6">
              <SourceFaceUpload
                targetCharacter={selectedCharacter}
                onUploadComplete={handleSourceFaceUpload}
                onError={handleError}
              />

              {/* Settings + Start button */}
              {sourceFace && (
                <div className="space-y-6 animate-fade-in">
                  <SettingsPanel settings={settings} onChange={setSettings} />
                  <div className="flex justify-center">
                    <button onClick={handleStartSwap} className="btn-primary text-lg px-10 py-4">
                      🚀 Start Face Swap
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 4 && jobId && (
            <ProcessingView
              jobId={jobId}
              onComplete={handleProcessingComplete}
              onError={handleError}
              onRetry={() => setStep(3)}
              onOpenLogs={() => setShowLogs(true)}
            />
          )}

          {step === 5 && jobId && (
            <ResultView jobId={jobId} onReset={handleReset} />
          )}
        </div>
      </main>

      {/* ─── Footer ─────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-900 py-4">
        <p className="text-center text-xs text-zinc-700">
          Face Swapper · Built with Next.js &amp; Tailwind CSS
        </p>
      </footer>

      {/* ─── Toast Notifications ────────────────────────────────── */}
      <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3 max-w-sm">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`
              animate-slide-up px-4 py-3 rounded-xl shadow-2xl backdrop-blur-sm
              border flex items-start gap-3 cursor-pointer
              ${
                toast.type === "error"
                  ? "bg-red-950/80 border-red-800/50 text-red-200"
                  : toast.type === "success"
                  ? "bg-green-950/80 border-green-800/50 text-green-200"
                  : "bg-blue-950/80 border-blue-800/50 text-blue-200"
              }
            `}
            onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
          >
            <span className="mt-0.5">
              {toast.type === "error" ? "❌" : toast.type === "success" ? "✅" : "ℹ️"}
            </span>
            <p className="text-sm">{toast.message}</p>
          </div>
        ))}
      </div>
      {/* ─── Logs & Diagnostics Modal ──────────────────────────── */}
      <LogsModal
        isOpen={showLogs}
        onClose={() => setShowLogs(false)}
        onLogsCleared={() => addToast("info", "All older log files cleared")}
      />
    </>
  );
}
