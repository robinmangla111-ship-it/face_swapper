"use client";

import React from "react";

const STEPS = [
  { num: 1, label: "Upload Video" },
  { num: 2, label: "Select Face" },
  { num: 3, label: "Source Face" },
  { num: 4, label: "Processing" },
  { num: 5, label: "Result" },
];

interface StepIndicatorProps {
  currentStep: number;
}

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <div className="flex items-center justify-between">
        {STEPS.map((step, idx) => {
          const isActive = step.num === currentStep;
          const isCompleted = step.num < currentStep;
          const isLast = idx === STEPS.length - 1;

          return (
            <React.Fragment key={step.num}>
              {/* Step circle + label */}
              <div className="flex flex-col items-center gap-2 relative">
                <div
                  className={`
                    w-10 h-10 rounded-full flex items-center justify-center
                    text-sm font-bold transition-all duration-500
                    ${
                      isCompleted
                        ? "bg-brand-500 text-white shadow-lg shadow-brand-500/30"
                        : isActive
                        ? "bg-brand-600 text-white ring-4 ring-brand-500/30 shadow-lg shadow-brand-600/40"
                        : "bg-zinc-800 text-zinc-500 border border-zinc-700"
                    }
                  `}
                >
                  {isCompleted ? (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    step.num
                  )}
                </div>
                <span
                  className={`
                    text-xs font-medium whitespace-nowrap transition-colors duration-300
                    ${isActive ? "text-brand-400" : isCompleted ? "text-zinc-400" : "text-zinc-600"}
                  `}
                >
                  {step.label}
                </span>
              </div>

              {/* Connector line */}
              {!isLast && (
                <div className="flex-1 h-0.5 mx-2 mb-6 relative overflow-hidden rounded-full">
                  <div className="absolute inset-0 bg-zinc-800" />
                  <div
                    className={`
                      absolute inset-y-0 left-0 bg-brand-500 transition-all duration-700 ease-out rounded-full
                      ${isCompleted ? "w-full" : "w-0"}
                    `}
                  />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
