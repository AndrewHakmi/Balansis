import { Play, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { PythonPlayground } from '../components/PythonPlayground';

export function Examples() {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const examples = [
    {
      title: "Basic Arithmetic",
      description: "Demonstrate perfect precision in basic mathematical operations",
      code: `from balansis import AbsoluteValue

# Traditional floating point issues
print(0.1 + 0.2)  # 0.30000000000000004

# Perfect precision with Balansis
a = AbsoluteValue(0.1)
b = AbsoluteValue(0.2)
result = a + b
print(result)  # Exactly 0.3

# More complex operations
x = AbsoluteValue("3.14159265359")
y = AbsoluteValue("2.71828182846")
print(x * y)  # Perfect multiplication
print(x / y)  # Perfect division`
    },
    {
      title: "Handling Infinity",
      description: "Work with infinite values using EternalRatio",
      code: `from balansis import AbsoluteValue, EternalRatio

# Create infinite values
pos_inf = EternalRatio.POSITIVE_INFINITY
neg_inf = EternalRatio.NEGATIVE_INFINITY
undefined = EternalRatio.UNDEFINED

# Operations with infinity
finite = AbsoluteValue(42)
print(pos_inf + finite)  # Positive infinity
print(pos_inf * neg_inf)  # Negative infinity
print(pos_inf / pos_inf)  # Undefined

# Check for infinity
if result.is_infinite():
    print("Result is infinite")
elif result.is_undefined():
    print("Result is undefined")`
    },
    {
      title: "Compensated Operations",
      description: "Use advanced compensation for high-precision computations",
      code: `from balansis import AbsoluteValue
from balansis.logic import Compensator

# Create compensator
comp = Compensator(precision_threshold=1e-15)

# Large sum with compensation
values = [AbsoluteValue(1/i) for i in range(1, 1000)]
compensated_sum = comp.compensated_sum(values)
print(f"Compensated sum: {compensated_sum}")

# Compensated multiplication
factors = [AbsoluteValue(1.1), AbsoluteValue(0.9), AbsoluteValue(1.05)]
result = comp.compensated_multiply(factors)
print(f"Compensated product: {result}")

# Check compensation effectiveness
print(f"Compensation applied: {comp.last_compensation_magnitude}")`
    },
    {
      title: "Scientific Computing",
      description: "Apply ACT to scientific and engineering calculations",
      code: `from balansis import AbsoluteValue
import math

# Physical constants with perfect precision
c = AbsoluteValue("299792458")  # Speed of light (m/s)
h = AbsoluteValue("6.62607015e-34")  # Planck constant
e = AbsoluteValue("1.602176634e-19")  # Elementary charge

# Energy calculation: E = hf
frequency = AbsoluteValue("5e14")  # 500 THz
energy = h * frequency
print(f"Photon energy: {energy} J")

# Relativistic calculations
mass = AbsoluteValue("9.1093837015e-31")  # Electron mass
rest_energy = mass * c * c
print(f"Electron rest energy: {rest_energy} J")

# Convert to eV
energy_eV = rest_energy / e
print(f"Electron rest energy: {energy_eV} eV")`
    },
    {
      title: "Financial Calculations",
      description: "Precise financial computations without rounding errors",
      code: `from balansis import AbsoluteValue

# Compound interest calculation
principal = AbsoluteValue("10000.00")
rate = AbsoluteValue("0.05")  # 5% annual rate
time = AbsoluteValue("10")    # 10 years

# A = P(1 + r)^t
one = AbsoluteValue("1")
base = one + rate
amount = principal * (base ** time)
print(f"Final amount: $" + str(amount))

# Monthly payments calculation
loan_amount = AbsoluteValue("250000")  # $250,000 loan
monthly_rate = AbsoluteValue("0.004")  # 4.8% annual / 12 months
months = AbsoluteValue("360")          # 30 years

# PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
factor = (one + monthly_rate) ** months
payment = loan_amount * (monthly_rate * factor) / (factor - one)
print(f"Monthly payment: $" + str(payment))`
    },
    {
      title: "Algebraic Structures",
      description: "Work with mathematical groups and fields",
      code: `from balansis.algebra import AbsoluteGroup, EternityField

# Create group with AbsoluteValue elements
group = AbsoluteGroup()
a = group.create_element("2.5")
b = group.create_element("3.7")

# Group operations
sum_ab = group.operate(a, b)
print(f"Group operation: {sum_ab}")

# Check group properties
identity = group.identity()
inverse_a = group.inverse(a)
print(f"a + (-a) = {group.operate(a, inverse_a)}")

# Work with eternity field
field = EternityField()
x = field.create_element("5.0")
y = field.create_element("2.0")

# Field operations
product = field.multiply(x, y)
quotient = field.divide(x, y)
print(f"Field multiplication: {product}")
print(f"Field division: {quotient}")`
    }
  ];

  const copyToClipboard = async (code: string, index: number) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  return (
    <div className="min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Examples
          </h1>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Explore practical examples of Balansis in action. From basic arithmetic 
            to advanced scientific computing, see how ACT transforms mathematical precision.
          </p>
        </div>

        <div className="space-y-8">
          {examples.map((example, index) => (
            <div key={index} className="bg-white/10 backdrop-blur-md rounded-xl overflow-hidden">
              <div className="p-6 border-b border-white/10">
                <h3 className="text-2xl font-bold text-white mb-2">{example.title}</h3>
                <p className="text-white/70">{example.description}</p>
              </div>
              <div className="relative">
                <div className="absolute top-4 right-4 flex space-x-2">
                  <button
                    onClick={() => copyToClipboard(example.code, index)}
                    className="bg-white/10 hover:bg-white/20 text-white p-2 rounded-lg transition-colors"
                    title="Copy code"
                  >
                    {copiedIndex === index ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                  <button
                    className="bg-accent-500 hover:bg-accent-600 text-white p-2 rounded-lg transition-colors"
                    title="Run in playground"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                </div>
                <pre className="bg-gray-900 p-6 overflow-x-auto">
                  <code className="text-sm text-white whitespace-pre">
                    {example.code}
                  </code>
                </pre>
              </div>
            </div>
          ))}
        </div>

        {/* Interactive Playground */}
        <div className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Interactive Playground
            </h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Experience Balansis in real-time with our WebAssembly-powered Python environment
            </p>
          </div>
          
          <PythonPlayground />
        </div>
      </div>
    </div>
  );
}