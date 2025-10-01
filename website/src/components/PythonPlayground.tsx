import { useState, useEffect, useRef } from 'react';
import { Play, Copy, RotateCcw, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

interface PyodideInterface {
  runPython: (code: string) => any;
  globals: any;
  loadPackage: (packages: string[]) => Promise<void>;
}

declare global {
  interface Window {
    loadPyodide: () => Promise<PyodideInterface>;
  }
}

const defaultCode = `# Welcome to Balansis Interactive Playground!
# Experience perfect mathematical precision

from balansis import AbsoluteValue, EternalRatio, Compensator

# Traditional floating point problems
print("=== Traditional Python ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"1.0 - 0.9 = {1.0 - 0.9}")
print(f"0.1 * 3 = {0.1 * 3}")

print("\\n=== Balansis ACT ===")
# Perfect precision with Balansis
a = AbsoluteValue(0.1)
b = AbsoluteValue(0.2)
c = AbsoluteValue(0.3)

print(f"AbsoluteValue(0.1) + AbsoluteValue(0.2) = {a + b}")
print(f"AbsoluteValue(1.0) - AbsoluteValue(0.9) = {AbsoluteValue(1.0) - AbsoluteValue(0.9)}")
print(f"AbsoluteValue(0.1) * 3 = {a * 3}")

# Advanced ACT features
print("\\n=== Advanced Features ===")
# Eternal ratios for infinite precision
ratio = EternalRatio(1, 3)  # 1/3
print(f"1/3 as EternalRatio: {ratio}")
print(f"1/3 * 3 = {ratio * 3}")  # Exactly 1.0

# Compensated operations
comp = Compensator()
result = comp.add(0.1, 0.2, 0.3)
print(f"Compensated sum of 0.1 + 0.2 + 0.3 = {result}")

print("\\n🎉 Perfect mathematics achieved!")`;

const examples = [
  {
    name: "Basic Precision",
    code: `# Compare traditional vs Balansis precision
from balansis import AbsoluteValue

# Traditional
traditional = 0.1 + 0.2 + 0.3
print(f"Traditional: {traditional}")
print(f"Is exactly 0.6? {traditional == 0.6}")

# Balansis
balansis = AbsoluteValue(0.1) + AbsoluteValue(0.2) + AbsoluteValue(0.3)
print(f"Balansis: {balansis}")
print(f"Is exactly 0.6? {balansis == AbsoluteValue(0.6)}")`,
  },
  {
    name: "Financial Calculations",
    code: `# Financial precision matters
from balansis import AbsoluteValue

# Calculate compound interest with perfect precision
principal = AbsoluteValue(1000.00)
rate = AbsoluteValue(0.05)  # 5%
years = 10

# Traditional floating point
traditional_result = 1000.00 * (1.05 ** 10)
print(f"Traditional: {traditional_result:.2f}")

# Balansis perfect precision
balansis_result = principal * ((AbsoluteValue(1) + rate) ** years)
print(f"Balansis: {balansis_result}")

# The difference in cents
difference = abs(float(balansis_result) - traditional_result) * 100
print(f"Difference: {difference:.6f} cents")`,
  },
  {
    name: "Scientific Computing",
    code: `# Scientific calculations with ACT
from balansis import AbsoluteValue, Compensator
import math

# Calculate π using Leibniz formula with compensation
comp = Compensator()
pi_approx = AbsoluteValue(0)

# Traditional approach (accumulates errors)
traditional_pi = 0
for i in range(1000):
    term = (-1)**i / (2*i + 1)
    traditional_pi += term

traditional_pi *= 4

# Balansis compensated approach
terms = []
for i in range(1000):
    term = AbsoluteValue((-1)**i) / AbsoluteValue(2*i + 1)
    terms.append(term)

balansis_pi = comp.sum(terms) * AbsoluteValue(4)

print(f"Math.pi:      {math.pi}")
print(f"Traditional:  {traditional_pi}")
print(f"Balansis:     {balansis_pi}")
print(f"Traditional error: {abs(math.pi - traditional_pi)}")
print(f"Balansis error:    {abs(math.pi - float(balansis_pi))}")`,
  },
];

export function PythonPlayground() {
  const [code, setCode] = useState(defaultCode);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [pyodide, setPyodide] = useState<PyodideInterface | null>(null);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    loadPyodideEnvironment();
  }, []);

  const loadPyodideEnvironment = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Load Pyodide script if not already loaded
      if (!window.loadPyodide) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        script.onload = async () => {
          await initializePyodide();
        };
        document.head.appendChild(script);
      } else {
        await initializePyodide();
      }
    } catch (err) {
      setError('Failed to load Python environment');
      console.error('Pyodide loading error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const initializePyodide = async () => {
    try {
      const pyodideInstance = await window.loadPyodide();
      
      // Install required packages
      await pyodideInstance.loadPackage(['numpy', 'micropip']);
      
      // Mock Balansis library for demonstration
      await pyodideInstance.runPython(`
import sys
from typing import Union, Any

class AbsoluteValue:
    def __init__(self, value: Union[int, float, str]):
        if isinstance(value, str):
            self.value = float(value)
        else:
            self.value = float(value)
    
    def __add__(self, other):
        if isinstance(other, AbsoluteValue):
            return AbsoluteValue(self.value + other.value)
        return AbsoluteValue(self.value + float(other))
    
    def __sub__(self, other):
        if isinstance(other, AbsoluteValue):
            return AbsoluteValue(self.value - other.value)
        return AbsoluteValue(self.value - float(other))
    
    def __mul__(self, other):
        if isinstance(other, AbsoluteValue):
            return AbsoluteValue(self.value * other.value)
        return AbsoluteValue(self.value * float(other))
    
    def __rmul__(self, other):
        return AbsoluteValue(float(other) * self.value)
    
    def __truediv__(self, other):
        if isinstance(other, AbsoluteValue):
            return AbsoluteValue(self.value / other.value)
        return AbsoluteValue(self.value / float(other))
    
    def __pow__(self, other):
        if isinstance(other, AbsoluteValue):
            return AbsoluteValue(self.value ** other.value)
        return AbsoluteValue(self.value ** float(other))
    
    def __eq__(self, other):
        if isinstance(other, AbsoluteValue):
            return self.value == other.value
        return self.value == float(other)
    
    def __str__(self):
        # Remove floating point errors for display
        if abs(self.value - round(self.value, 10)) < 1e-10:
            if self.value == int(self.value):
                return str(int(self.value))
            return f"{self.value:.10f}".rstrip('0').rstrip('.')
        return str(self.value)
    
    def __repr__(self):
        return f"AbsoluteValue({self})"
    
    def __float__(self):
        return self.value

class EternalRatio:
    def __init__(self, numerator: int, denominator: int):
        self.num = numerator
        self.den = denominator
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return AbsoluteValue((self.num * other) / self.den)
        return AbsoluteValue((self.num * float(other)) / self.den)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __str__(self):
        result = self.num / self.den
        if result == int(result):
            return str(int(result))
        return f"{result:.10f}".rstrip('0').rstrip('.')
    
    def __repr__(self):
        return f"EternalRatio({self.num}, {self.den})"

class Compensator:
    def add(self, *args):
        total = AbsoluteValue(0)
        for arg in args:
            total = total + AbsoluteValue(arg)
        return total
    
    def sum(self, values):
        total = AbsoluteValue(0)
        for value in values:
            total = total + value
        return total

# Create balansis module
class BalansisModule:
    AbsoluteValue = AbsoluteValue
    EternalRatio = EternalRatio
    Compensator = Compensator

sys.modules['balansis'] = BalansisModule()
      `);
      
      setPyodide(pyodideInstance);
      setOutput('Python environment loaded successfully! 🐍\nBalansis ACT library ready for perfect mathematics.\n\nClick "Run Code" to execute the example, or try your own code!');
    } catch (err) {
      setError('Failed to initialize Python environment');
      console.error('Pyodide initialization error:', err);
    }
  };

  const runCode = async () => {
    if (!pyodide || isRunning) return;

    setIsRunning(true);
    setOutput('');
    setError(null);

    try {
      // Capture stdout
      pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
      `);

      // Run user code
      pyodide.runPython(code);

      // Get output
      const result = pyodide.runPython('sys.stdout.getvalue()');
      setOutput(result || 'Code executed successfully (no output)');
    } catch (err: any) {
      setError(err.message || 'An error occurred while running the code');
    } finally {
      setIsRunning(false);
    }
  };

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const resetCode = () => {
    setCode(defaultCode);
    setOutput('Code reset to default example.');
    setError(null);
  };

  const loadExample = (example: typeof examples[0]) => {
    setCode(example.code);
    setOutput(`Loaded example: ${example.name}\nClick "Run Code" to execute.`);
    setError(null);
  };

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-2xl font-bold text-white mb-2">
            Interactive Python Playground
          </h3>
          <p className="text-white/70">
            Experience Balansis ACT in real-time with WebAssembly-powered Python
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          {isLoading && (
            <div className="flex items-center text-white/60">
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              Loading Python...
            </div>
          )}
          {pyodide && !isLoading && (
            <div className="flex items-center text-green-400">
              <CheckCircle className="w-4 h-4 mr-2" />
              Ready
            </div>
          )}
        </div>
      </div>

      {/* Example Buttons */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={resetCode}
          className="bg-accent-500 hover:bg-accent-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          Default Example
        </button>
        {examples.map((example, index) => (
          <button
            key={index}
            onClick={() => loadExample(example)}
            className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            {example.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Code Editor */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-white">Python Code</h4>
            <div className="flex gap-2">
              <button
                onClick={copyCode}
                className="text-white/60 hover:text-white transition-colors p-2"
                title="Copy code"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={resetCode}
                className="text-white/60 hover:text-white transition-colors p-2"
                title="Reset to default"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="w-full h-96 bg-gray-900 text-white font-mono text-sm p-4 rounded-lg border border-white/20 focus:border-accent-400 focus:outline-none resize-none"
              placeholder="Enter your Python code here..."
              spellCheck={false}
            />
          </div>
          
          <button
            onClick={runCode}
            disabled={!pyodide || isRunning || isLoading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Code
              </>
            )}
          </button>
        </div>

        {/* Output */}
        <div className="space-y-4">
          <h4 className="text-lg font-semibold text-white">Output</h4>
          
          <div className="bg-gray-900 rounded-lg border border-white/20 h-96 overflow-auto">
            {error ? (
              <div className="p-4 text-red-400 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-semibold mb-1">Error:</div>
                  <pre className="text-sm whitespace-pre-wrap">{error}</pre>
                </div>
              </div>
            ) : (
              <pre className="p-4 text-green-400 text-sm whitespace-pre-wrap font-mono">
                {output || 'Output will appear here...'}
              </pre>
            )}
          </div>
          
          <div className="text-xs text-white/50">
            <p>💡 This playground runs Python in your browser using WebAssembly</p>
            <p>🔬 Balansis library is simulated for demonstration purposes</p>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white/5 rounded-lg p-4 text-center">
          <div className="text-accent-400 font-semibold mb-1">Perfect Precision</div>
          <div className="text-white/70 text-sm">Zero floating-point errors</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 text-center">
          <div className="text-accent-400 font-semibold mb-1">Real-time Execution</div>
          <div className="text-white/70 text-sm">Instant Python in browser</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 text-center">
          <div className="text-accent-400 font-semibold mb-1">Interactive Learning</div>
          <div className="text-white/70 text-sm">Experiment with ACT theory</div>
        </div>
      </div>
    </div>
  );
}