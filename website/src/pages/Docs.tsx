import { Book, Code, FileText, Search } from 'lucide-react';

export function Docs() {
  return (
    <div className="min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Documentation
          </h1>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Complete guide to using Balansis and implementing Absolute Compensation Theory 
            in your mathematical computations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 sticky top-24">
              <div className="flex items-center space-x-2 mb-4">
                <Search className="w-5 h-5 text-white/70" />
                <input
                  type="text"
                  placeholder="Search docs..."
                  className="bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white placeholder-white/50 text-sm flex-1"
                />
              </div>
              <nav className="space-y-2">
                <a href="#getting-started" className="block text-accent-400 hover:text-accent-300 text-sm font-medium">
                  Getting Started
                </a>
                <a href="#installation" className="block text-white/70 hover:text-white text-sm pl-4">
                  Installation
                </a>
                <a href="#quick-start" className="block text-white/70 hover:text-white text-sm pl-4">
                  Quick Start
                </a>
                <a href="#core-concepts" className="block text-white/70 hover:text-white text-sm font-medium">
                  Core Concepts
                </a>
                <a href="#absolute-value" className="block text-white/70 hover:text-white text-sm pl-4">
                  AbsoluteValue
                </a>
                <a href="#eternal-ratio" className="block text-white/70 hover:text-white text-sm pl-4">
                  EternalRatio
                </a>
                <a href="#compensator" className="block text-white/70 hover:text-white text-sm pl-4">
                  Compensator
                </a>
                <a href="#api-reference" className="block text-white/70 hover:text-white text-sm font-medium">
                  API Reference
                </a>
                <a href="#examples" className="block text-white/70 hover:text-white text-sm font-medium">
                  Examples
                </a>
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-8">
              <section id="getting-started" className="mb-12">
                <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                  <Book className="w-8 h-8 mr-3 text-accent-400" />
                  Getting Started
                </h2>
                
                <div id="installation" className="mb-8">
                  <h3 className="text-2xl font-semibold text-white mb-4">Installation</h3>
                  <p className="text-white/70 mb-4">
                    Install Balansis using pip:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4 mb-4">
                    <code className="text-accent-400 font-mono">pip install balansis</code>
                  </div>
                  <p className="text-white/70 mb-4">
                    Or install from source:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4">
                    <code className="text-accent-400 font-mono text-sm">
                      git clone https://github.com/balansis/balansis.git<br />
                      cd balansis<br />
                      pip install -e .
                    </code>
                  </div>
                </div>

                <div id="quick-start" className="mb-8">
                  <h3 className="text-2xl font-semibold text-white mb-4">Quick Start</h3>
                  <p className="text-white/70 mb-4">
                    Here's a simple example to get you started with Balansis:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4">
                    <pre className="text-sm">
                      <code className="text-white">
{`from balansis import AbsoluteValue, EternalRatio

# Create precise values
a = AbsoluteValue(0.1)
b = AbsoluteValue(0.2)

# Perfect arithmetic
result = a + b
print(result)  # Exactly 0.3

# Handle infinity
infinity = EternalRatio.POSITIVE_INFINITY
finite = AbsoluteValue(42)
print(infinity + finite)  # Still infinity`}
                      </code>
                    </pre>
                  </div>
                </div>
              </section>

              <section id="core-concepts" className="mb-12">
                <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                  <Code className="w-8 h-8 mr-3 text-accent-400" />
                  Core Concepts
                </h2>

                <div id="absolute-value" className="mb-8">
                  <h3 className="text-2xl font-semibold text-white mb-4">AbsoluteValue</h3>
                  <p className="text-white/70 mb-4">
                    The foundation of ACT, representing numbers with perfect precision:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm">
                      <code className="text-white">
{`# Create AbsoluteValue instances
val1 = AbsoluteValue(3.14159)
val2 = AbsoluteValue("2.71828")  # From string for exact precision
val3 = AbsoluteValue(Decimal("1.41421"))  # From Decimal

# All arithmetic operations preserve precision
sum_val = val1 + val2
product = val1 * val2
quotient = val1 / val2`}
                      </code>
                    </pre>
                  </div>
                </div>

                <div id="eternal-ratio" className="mb-8">
                  <h3 className="text-2xl font-semibold text-white mb-4">EternalRatio</h3>
                  <p className="text-white/70 mb-4">
                    Handles infinite and undefined values mathematically:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm">
                      <code className="text-white">
{`# Infinity operations
pos_inf = EternalRatio.POSITIVE_INFINITY
neg_inf = EternalRatio.NEGATIVE_INFINITY
undefined = EternalRatio.UNDEFINED

# Mathematical operations with infinity
result1 = pos_inf + AbsoluteValue(100)  # Still positive infinity
result2 = pos_inf * neg_inf  # Negative infinity
result3 = pos_inf / pos_inf  # Undefined`}
                      </code>
                    </pre>
                  </div>
                </div>

                <div id="compensator" className="mb-8">
                  <h3 className="text-2xl font-semibold text-white mb-4">Compensator</h3>
                  <p className="text-white/70 mb-4">
                    Advanced error compensation for complex operations:
                  </p>
                  <div className="bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm">
                      <code className="text-white">
{`from balansis.logic import Compensator

# Create compensator for high-precision operations
comp = Compensator(precision_threshold=1e-15)

# Compensated operations
values = [AbsoluteValue(x) for x in [0.1, 0.2, 0.3, 0.4]]
compensated_sum = comp.compensated_sum(values)

# Maintains precision even with many operations
large_computation = comp.compensated_multiply(values)`}
                      </code>
                    </pre>
                  </div>
                </div>
              </section>

              <section id="api-reference">
                <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                  <FileText className="w-8 h-8 mr-3 text-accent-400" />
                  API Reference
                </h2>
                <p className="text-white/70 mb-4">
                  For complete API documentation, including all classes, methods, and parameters, 
                  please refer to our comprehensive API reference.
                </p>
                <div className="bg-accent-500/20 border border-accent-500/30 rounded-lg p-4">
                  <p className="text-accent-200">
                    📚 <strong>Coming Soon:</strong> Interactive API browser with live examples and type information.
                  </p>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}