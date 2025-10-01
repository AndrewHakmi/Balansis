import { Brain, Lightbulb, Target, Zap } from 'lucide-react';
import { MathBlock, MathInline } from '../components/MathRenderer';

export function Theory() {
  return (
    <div className="min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Absolute Compensation Theory
          </h1>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Discover the mathematical foundation that revolutionizes numerical computation, 
            eliminating floating-point errors and enabling perfect mathematical precision.
          </p>
        </div>

        <div className="space-y-16">
          {/* Introduction */}
          <section className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
              <Brain className="w-8 h-8 mr-3 text-accent-400" />
              What is ACT?
            </h2>
            <div className="prose prose-lg text-white/80 max-w-none">
              <p className="mb-6">
                Absolute Compensation Theory (ACT) is a revolutionary mathematical framework that addresses 
                the fundamental limitations of floating-point arithmetic in digital computation. Unlike 
                traditional approaches that accept approximation errors as inevitable, ACT provides a 
                systematic method for achieving perfect numerical precision.
              </p>
              <p className="mb-6">
                The theory is built on three core principles:
              </p>
              <ul className="list-disc list-inside space-y-2 mb-6">
                <li><strong>Absolute Representation:</strong> Numbers are represented in their exact mathematical form</li>
                <li><strong>Compensation Logic:</strong> Systematic error detection and correction mechanisms</li>
                <li><strong>Eternity Handling:</strong> Proper mathematical treatment of infinite and undefined values</li>
              </ul>
            </div>
          </section>

          {/* Mathematical Foundation */}
          <section className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
              <Target className="w-8 h-8 mr-3 text-accent-400" />
              Mathematical Foundation
            </h2>
            <div className="space-y-6">
              <div>
                <h4 className="text-xl font-semibold text-white mb-3">AbsoluteValue Definition</h4>
                <div className="bg-gray-900 rounded-lg p-6">
                  <MathBlock>
                    {`\\abs{x} = x \\oplus \\comp{\\epsilon(x)}`}
                  </MathBlock>
                </div>
                <p className="text-white/70 mt-2">
                  Where <MathInline>{`\\oplus`}</MathInline> represents the compensated addition operation that eliminates floating-point errors,
                  and <MathInline>{`\\epsilon(x)`}</MathInline> is the representation error.
                </p>
              </div>
              
              <div>
                <h4 className="text-xl font-semibold text-white mb-3">Compensation Function</h4>
                <div className="bg-gray-900 rounded-lg p-6">
                  <MathBlock>
                    {`\\comp{x, y} = (x + y) - ((x + y) - x - y)`}
                  </MathBlock>
                </div>
                <p className="text-white/70 mt-2">
                  The compensation function captures and corrects the error <MathInline>{`\\epsilon`}</MathInline> introduced by floating-point arithmetic.
                </p>
              </div>

              <div>
                <h4 className="text-xl font-semibold text-white mb-3">Eternal Ratio</h4>
                <div className="bg-gray-900 rounded-lg p-6">
                  <MathBlock>
                    {`\\eternal{\\frac{p}{q}} = \\lim_{n \\to \\infty} \\frac{p \\cdot n}{q \\cdot n}`}
                  </MathBlock>
                </div>
                <p className="text-white/70 mt-2">
                  Eternal ratios maintain perfect precision for rational numbers, avoiding the limitations of decimal representation.
                </p>
              </div>
            </div>
          </section>

          {/* Core Concepts */}
          <section className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
              <Lightbulb className="w-8 h-8 mr-3 text-accent-400" />
              Core Concepts
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white/5 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-3">Exact Arithmetic</h3>
                <p className="text-white/70 text-sm">
                  All operations maintain mathematical exactness through rational number 
                  representation and symbolic computation techniques.
                </p>
              </div>
              <div className="bg-white/5 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-3">Error Compensation</h3>
                <p className="text-white/70 text-sm">
                  Systematic detection and correction of computational errors using 
                  advanced compensation algorithms.
                </p>
              </div>
              <div className="bg-white/5 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-white mb-3">Infinity Handling</h3>
                <p className="text-white/70 text-sm">
                  Proper mathematical treatment of infinite values through the 
                  EternalRatio system.
                </p>
              </div>
            </div>
          </section>

          {/* Axioms */}
          <section className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
              <Zap className="w-8 h-8 mr-3 text-accent-400" />
              ACT Axioms
            </h2>
            <div className="space-y-6">
              <div className="border-l-4 border-accent-400 pl-6">
                <h3 className="text-xl font-semibold text-white mb-2">Axiom 1: Compensation</h3>
                <p className="text-white/70">
                  For any operation that introduces computational error, there exists a 
                  compensation value that restores mathematical exactness.
                </p>
                <div className="bg-gray-900 rounded-lg p-3 mt-3">
                  <code className="text-accent-400 text-sm">
                    ∀ op, a, b: exact(op(a,b)) = computed(op(a,b)) + compensation(op,a,b)
                  </code>
                </div>
              </div>
              
              <div className="border-l-4 border-accent-400 pl-6">
                <h3 className="text-xl font-semibold text-white mb-2">Axiom 2: Stability</h3>
                <p className="text-white/70">
                  Compensated operations maintain numerical stability across all scales 
                  of computation, from infinitesimal to infinite values.
                </p>
                <div className="bg-gray-900 rounded-lg p-3 mt-3">
                  <code className="text-accent-400 text-sm">
                    ∀ scale ∈ ℝ: stability(compensated_op(scale × input)) = stability(input)
                  </code>
                </div>
              </div>
              
              <div className="border-l-4 border-accent-400 pl-6">
                <h3 className="text-xl font-semibold text-white mb-2">Axiom 3: Eternity</h3>
                <p className="text-white/70">
                  Infinite and undefined values are treated as first-class mathematical 
                  objects with well-defined operational semantics.
                </p>
                <div className="bg-gray-900 rounded-lg p-3 mt-3">
                  <code className="text-accent-400 text-sm">
                    ∞ + finite = ∞, ∞ × ∞ = ∞, ∞ ÷ ∞ = undefined
                  </code>
                </div>
              </div>
            </div>
          </section>

          {/* Applications */}
          <section className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6">Applications & Impact</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Scientific Computing</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• Quantum mechanics calculations</li>
                  <li>• Climate modeling precision</li>
                  <li>• Astronomical computations</li>
                  <li>• Particle physics simulations</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Financial Systems</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• High-frequency trading</li>
                  <li>• Risk assessment models</li>
                  <li>• Cryptocurrency precision</li>
                  <li>• Actuarial calculations</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Engineering</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• Structural analysis</li>
                  <li>• Control systems</li>
                  <li>• Signal processing</li>
                  <li>• Optimization algorithms</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Machine Learning</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• Gradient descent precision</li>
                  <li>• Neural network training</li>
                  <li>• Numerical optimization</li>
                  <li>• Statistical computations</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Research */}
          <section className="bg-gradient-to-r from-accent-500/20 to-primary-500/20 backdrop-blur-md rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-6">Ongoing Research</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Current Areas</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• Quantum-resistant compensation algorithms</li>
                  <li>• Distributed ACT implementations</li>
                  <li>• Hardware acceleration techniques</li>
                  <li>• Formal verification methods</li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-4">Future Directions</h3>
                <ul className="space-y-2 text-white/70">
                  <li>• Integration with quantum computing</li>
                  <li>• Real-time compensation systems</li>
                  <li>• Adaptive precision algorithms</li>
                  <li>• Cross-platform standardization</li>
                </ul>
              </div>
            </div>
            <div className="mt-8 text-center">
              <p className="text-white/70 mb-4">
                Interested in contributing to ACT research?
              </p>
              <button className="bg-accent-500 hover:bg-accent-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
                Join Research Community
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}