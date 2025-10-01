import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';

interface MathStep {
  operation: string;
  traditional: string;
  balansis: string;
  explanation: string;
}

const mathSteps: MathStep[] = [
  {
    operation: "0.1 + 0.2",
    traditional: "0.30000000000000004",
    balansis: "0.3",
    explanation: "Basic decimal addition"
  },
  {
    operation: "0.1 + 0.2 + 0.3",
    traditional: "0.6000000000000001",
    balansis: "0.6",
    explanation: "Accumulated floating-point errors"
  },
  {
    operation: "1.0 - 0.9",
    traditional: "0.09999999999999998",
    balansis: "0.1",
    explanation: "Subtraction precision loss"
  },
  {
    operation: "0.1 * 3",
    traditional: "0.30000000000000004",
    balansis: "0.3",
    explanation: "Multiplication with decimals"
  },
  {
    operation: "1/3 + 1/3 + 1/3",
    traditional: "0.9999999999999999",
    balansis: "1.0",
    explanation: "Infinite precision fractions"
  }
];

export function AnimatedMathDemo() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [animationPhase, setAnimationPhase] = useState<'operation' | 'traditional' | 'balansis' | 'complete'>('operation');

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setAnimationPhase(prev => {
        switch (prev) {
          case 'operation':
            return 'traditional';
          case 'traditional':
            return 'balansis';
          case 'balansis':
            return 'complete';
          case 'complete':
            // Move to next step after a pause
            setTimeout(() => {
              setCurrentStep(prev => (prev + 1) % mathSteps.length);
              setAnimationPhase('operation');
            }, 1500);
            return 'complete';
          default:
            return 'operation';
        }
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [isPlaying]);

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const reset = () => {
    setCurrentStep(0);
    setAnimationPhase('operation');
    setIsPlaying(true);
  };

  const currentMathStep = mathSteps[currentStep];

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <h3 className="text-2xl font-bold text-white">
          Live Mathematical Precision Demo
        </h3>
        <div className="flex gap-3">
          <button
            onClick={togglePlayPause}
            className="bg-accent-500 hover:bg-accent-600 text-white p-3 rounded-lg transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          <button
            onClick={reset}
            className="bg-white/20 hover:bg-white/30 text-white p-3 rounded-lg transition-colors"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Operation Display */}
        <div className="space-y-6">
          <div className="text-center">
            <div className="text-sm text-white/60 mb-2">Computing:</div>
            <div className={`text-3xl font-mono font-bold transition-all duration-500 ${
              animationPhase === 'operation' ? 'text-accent-400 scale-110' : 'text-white'
            }`}>
              {currentMathStep.operation}
            </div>
            <div className="text-sm text-white/70 mt-2">
              {currentMathStep.explanation}
            </div>
          </div>

          {/* Progress Indicator */}
          <div className="flex justify-center space-x-2">
            {mathSteps.map((_, index) => (
              <div
                key={index}
                className={`w-3 h-3 rounded-full transition-colors duration-300 ${
                  index === currentStep ? 'bg-accent-400' : 'bg-white/30'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Results Comparison */}
        <div className="space-y-4">
          {/* Traditional Result */}
          <div className={`bg-red-500/20 border-2 rounded-xl p-6 transition-all duration-500 ${
            animationPhase === 'traditional' ? 'border-red-400 scale-105 shadow-lg shadow-red-500/20' : 'border-red-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-red-300 font-semibold">Traditional Float</span>
              <span className="text-xs text-red-400 bg-red-500/20 px-2 py-1 rounded">
                IMPRECISE
              </span>
            </div>
            <div className={`font-mono text-lg transition-all duration-500 ${
              animationPhase === 'traditional' ? 'text-red-200 font-bold' : 'text-red-300'
            }`}>
              {animationPhase === 'operation' ? '...' : currentMathStep.traditional}
            </div>
          </div>

          {/* Balansis Result */}
          <div className={`bg-green-500/20 border-2 rounded-xl p-6 transition-all duration-500 ${
            animationPhase === 'balansis' ? 'border-green-400 scale-105 shadow-lg shadow-green-500/20' : 'border-green-500/30'
          }`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-green-300 font-semibold">Balansis ACT</span>
              <span className="text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded">
                PERFECT
              </span>
            </div>
            <div className={`font-mono text-lg transition-all duration-500 ${
              animationPhase === 'balansis' || animationPhase === 'complete' ? 'text-green-200 font-bold' : 'text-green-300'
            }`}>
              {animationPhase === 'operation' || animationPhase === 'traditional' ? '...' : currentMathStep.balansis}
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-8 p-6 bg-white/5 rounded-xl">
        <div className="text-center">
          <div className="text-white/60 text-sm mb-2">Why the difference?</div>
          <div className="text-white/90 max-w-2xl mx-auto">
            Traditional floating-point arithmetic suffers from representation errors and accumulated precision loss. 
            Balansis uses <span className="text-accent-400 font-semibold">Absolute Compensation Theory</span> to 
            maintain perfect mathematical precision through advanced error compensation algorithms.
          </div>
        </div>
      </div>

      {/* Mathematical Visualization */}
      <div className="mt-6 flex justify-center">
        <div className="flex items-center space-x-4 text-sm text-white/70">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full"></div>
            <span>Floating Point Errors</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span>ACT Precision</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-accent-400 rounded-full animate-pulse"></div>
            <span>Active Computation</span>
          </div>
        </div>
      </div>
    </div>
  );
}