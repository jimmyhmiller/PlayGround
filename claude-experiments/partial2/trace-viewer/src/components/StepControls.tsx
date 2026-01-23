'use client';

import React from 'react';

interface StepControlsProps {
  currentStep: number;
  totalSteps: number;
  onStepChange: (step: number) => void;
}

export function StepControls({ currentStep, totalSteps, onStepChange }: StepControlsProps) {
  return (
    <div className="border-t border-gray-700 p-4">
      <div className="flex items-center gap-4">
        {/* Navigation buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onStepChange(0)}
            disabled={currentStep === 0}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            title="First (Home)"
          >
            ⏮
          </button>
          <button
            onClick={() => onStepChange(currentStep - 1)}
            disabled={currentStep === 0}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            title="Previous (←)"
          >
            ◀
          </button>
          <button
            onClick={() => onStepChange(currentStep + 1)}
            disabled={currentStep >= totalSteps - 1}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            title="Next (→)"
          >
            ▶
          </button>
          <button
            onClick={() => onStepChange(totalSteps - 1)}
            disabled={currentStep >= totalSteps - 1}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            title="Last (End)"
          >
            ⏭
          </button>
        </div>

        {/* Progress bar */}
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={totalSteps - 1}
            value={currentStep}
            onChange={(e) => onStepChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Step counter */}
        <div className="text-sm text-gray-400 min-w-[80px] text-right">
          {currentStep + 1} / {totalSteps}
        </div>
      </div>
    </div>
  );
}
