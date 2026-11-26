"use client";

import React, { useState } from "react";

export interface SwapSettings {
  enhance: boolean;
  numWorkers: number | null;
  falKey?: string;
}

interface SettingsPanelProps {
  settings: SwapSettings;
  onChange: (settings: SwapSettings) => void;
}

export default function SettingsPanel({ settings, onChange }: SettingsPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors mx-auto"
      >
        <svg
          className={`w-4 h-4 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
        Settings
      </button>

      {open && (
        <div className="mt-3 card p-5 animate-slide-up space-y-5">
          {/* Enhance toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-zinc-200">Face Enhancement</p>
              <p className="text-xs text-zinc-500 mt-0.5">
                Apply GFPGAN enhancement to improve face quality
              </p>
            </div>
            <button
              onClick={() => onChange({ ...settings, enhance: !settings.enhance })}
              className={`
                relative w-12 h-6 rounded-full transition-colors duration-200
                ${settings.enhance ? "bg-brand-600" : "bg-zinc-700"}
              `}
            >
              <div
                className={`
                  absolute top-0.5 w-5 h-5 rounded-full bg-white shadow-md
                  transition-transform duration-200
                  ${settings.enhance ? "translate-x-6" : "translate-x-0.5"}
                `}
              />
            </button>
          </div>

          {/* Workers slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-zinc-200">Workers</p>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Number of parallel processing workers (auto = system default)
                </p>
              </div>
              <span className="text-sm font-mono text-brand-400 min-w-[3ch] text-right">
                {settings.numWorkers === null ? "Auto" : settings.numWorkers}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-zinc-600">Auto</span>
              <input
                type="range"
                min={0}
                max={16}
                value={settings.numWorkers ?? 0}
                onChange={(e) => {
                  const v = parseInt(e.target.value);
                  onChange({ ...settings, numWorkers: v === 0 ? null : v });
                }}
                className="flex-1 h-1.5 bg-zinc-700 rounded-full appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-brand-500 [&::-webkit-slider-thumb]:shadow-md
                  [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <span className="text-xs text-zinc-600">16</span>
            </div>
          </div>

          {/* Optional Cloud AI Key for Full-Body Character Diffusion */}
          <div className="space-y-2 pt-2 border-t border-zinc-800/80">
            <div>
              <p className="text-sm font-medium text-zinc-200">
                Cloud AI Key (Optional — for 🧍 Full-Body Diffusion)
              </p>
              <p className="text-xs text-zinc-500 mt-0.5">
                Paste a <span className="text-zinc-300">fal.ai</span> API key for photorealistic full-body character replacement (Wan-Animate). Leave blank for offline local CPU mode.
              </p>
            </div>
            <input
              type="password"
              value={settings.falKey || ""}
              onChange={(e) => onChange({ ...settings, falKey: e.target.value })}
              placeholder="fal_..."
              className="w-full px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-700/80 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>
      )}
    </div>
  );
}
