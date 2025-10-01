import { Package, Terminal, Copy, Check, ExternalLink } from 'lucide-react';
import { useState } from 'react';

export function Download() {
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);

  const copyToClipboard = async (text: string, command: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCommand(command);
      setTimeout(() => setCopiedCommand(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Download Balansis
          </h1>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Get started with Balansis and experience the power of Absolute Compensation Theory 
            in your mathematical computations.
          </p>
        </div>

        {/* Installation Methods */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {/* PyPI Installation */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <div className="flex items-center mb-6">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mr-4">
                <Package className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-2xl font-semibold text-white">PyPI (Recommended)</h3>
                <p className="text-white/70">Install the latest stable version</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-gray-900 rounded-lg p-4 flex items-center justify-between">
                <code className="text-accent-400 font-mono">pip install balansis</code>
                <button
                  onClick={() => copyToClipboard('pip install balansis', 'pip')}
                  className="text-white/60 hover:text-white transition-colors"
                >
                  {copiedCommand === 'pip' ? (
                    <Check className="w-5 h-5 text-green-400" />
                  ) : (
                    <Copy className="w-5 h-5" />
                  )}
                </button>
              </div>
              
              <div className="text-sm text-white/70">
                <p className="mb-2"><strong>Requirements:</strong></p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Python 3.8 or higher</li>
                  <li>NumPy ≥ 1.20.0</li>
                  <li>Decimal (built-in)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Development Installation */}
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-8">
            <div className="flex items-center mb-6">
              <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center mr-4">
                <Terminal className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-2xl font-semibold text-white">Development</h3>
                <p className="text-white/70">Install from source for latest features</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <code className="text-accent-400 font-mono text-sm">git clone https://github.com/balansis/balansis.git</code>
                  <button
                    onClick={() => copyToClipboard('git clone https://github.com/balansis/balansis.git', 'git')}
                    className="text-white/60 hover:text-white transition-colors"
                  >
                    {copiedCommand === 'git' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <code className="text-accent-400 font-mono text-sm">cd balansis</code>
                  <button
                    onClick={() => copyToClipboard('cd balansis', 'cd')}
                    className="text-white/60 hover:text-white transition-colors"
                  >
                    {copiedCommand === 'cd' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <code className="text-accent-400 font-mono text-sm">pip install -e .</code>
                  <button
                    onClick={() => copyToClipboard('pip install -e .', 'dev')}
                    className="text-white/60 hover:text-white transition-colors"
                  >
                    {copiedCommand === 'dev' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
              
              <div className="text-sm text-white/70">
                <p className="mb-2"><strong>Additional dev dependencies:</strong></p>
                <div className="bg-gray-900 rounded-lg p-2">
                  <code className="text-accent-400 font-mono text-xs">pip install -e ".[dev]"</code>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Version Information */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-8 mb-16">
          <h3 className="text-2xl font-semibold text-white mb-6">Version Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-400 mb-2">v1.2.3</div>
              <div className="text-white/70 text-sm">Latest Stable</div>
              <div className="text-white/50 text-xs mt-1">Released: Feb 15, 2024</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-primary-400 mb-2">v1.3.0-beta</div>
              <div className="text-white/70 text-sm">Beta Release</div>
              <div className="text-white/50 text-xs mt-1">Released: Feb 28, 2024</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">v1.3.0-dev</div>
              <div className="text-white/70 text-sm">Development</div>
              <div className="text-white/50 text-xs mt-1">Updated: Daily</div>
            </div>
          </div>
        </div>

        {/* Quick Start */}
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-8 mb-16">
          <h3 className="text-2xl font-semibold text-white mb-6">Quick Start</h3>
          <p className="text-white/70 mb-6">
            Once installed, verify your installation and run your first ACT computation:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-6">
            <pre className="text-sm">
              <code className="text-white">
{`# Verify installation
python -c "import balansis; print(balansis.__version__)"

# Your first ACT computation
from balansis import AbsoluteValue

# Traditional floating point error
print(0.1 + 0.2)  # 0.30000000000000004

# Perfect precision with Balansis
a = AbsoluteValue(0.1)
b = AbsoluteValue(0.2)
result = a + b
print(result)  # Exactly 0.3

print("Welcome to perfect mathematics!")`}
              </code>
            </pre>
          </div>
        </div>

        {/* Alternative Downloads */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Package Managers</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between bg-white/5 rounded-lg p-3">
                <div>
                  <div className="text-white font-medium">Conda</div>
                  <div className="text-white/60 text-sm">conda install -c conda-forge balansis</div>
                </div>
                <button
                  onClick={() => copyToClipboard('conda install -c conda-forge balansis', 'conda')}
                  className="text-white/60 hover:text-white transition-colors"
                >
                  {copiedCommand === 'conda' ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
              
              <div className="flex items-center justify-between bg-white/5 rounded-lg p-3">
                <div>
                  <div className="text-white font-medium">Poetry</div>
                  <div className="text-white/60 text-sm">poetry add balansis</div>
                </div>
                <button
                  onClick={() => copyToClipboard('poetry add balansis', 'poetry')}
                  className="text-white/60 hover:text-white transition-colors"
                >
                  {copiedCommand === 'poetry' ? (
                    <Check className="w-4 h-4 text-green-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
            <h3 className="text-xl font-semibold text-white mb-4">Resources</h3>
            <div className="space-y-3">
              <a
                href="https://github.com/balansis/balansis/releases"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between bg-white/5 rounded-lg p-3 hover:bg-white/10 transition-colors"
              >
                <div>
                  <div className="text-white font-medium">Release Notes</div>
                  <div className="text-white/60 text-sm">View changelog and updates</div>
                </div>
                <ExternalLink className="w-4 h-4 text-white/60" />
              </a>
              
              <a
                href="https://pypi.org/project/balansis/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-between bg-white/5 rounded-lg p-3 hover:bg-white/10 transition-colors"
              >
                <div>
                  <div className="text-white font-medium">PyPI Package</div>
                  <div className="text-white/60 text-sm">Official Python package</div>
                </div>
                <ExternalLink className="w-4 h-4 text-white/60" />
              </a>
              
              <a
                href="/docs"
                className="flex items-center justify-between bg-white/5 rounded-lg p-3 hover:bg-white/10 transition-colors"
              >
                <div>
                  <div className="text-white font-medium">Documentation</div>
                  <div className="text-white/60 text-sm">Complete usage guide</div>
                </div>
                <ExternalLink className="w-4 h-4 text-white/60" />
              </a>
            </div>
          </div>
        </div>

        {/* Support */}
        <div className="bg-gradient-to-r from-accent-500/20 to-primary-500/20 backdrop-blur-md rounded-xl p-8 text-center">
          <h3 className="text-2xl font-bold text-white mb-4">Need Help?</h3>
          <p className="text-white/70 mb-6">
            Having trouble with installation or getting started? Our community is here to help.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-accent-500 hover:bg-accent-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              Join Discord
            </button>
            <button className="border-2 border-white/30 hover:border-white/50 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              GitHub Issues
            </button>
            <button className="border-2 border-white/30 hover:border-white/50 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              View Examples
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}