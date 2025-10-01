import { useState, useEffect } from 'react';
import { ArrowRight, Calculator, Infinity, Zap, Shield, Code, Users, Star, GitFork, Download } from 'lucide-react';
import { AnimatedMathDemo } from '../components/AnimatedMathDemo';
import { Link } from 'react-router-dom';
import { fetchGitHubStats, GitHubStats } from '../lib/github';

export default function Home() {
  const [stats, setStats] = useState<GitHubStats>({
    totalStars: 0,
    totalForks: 0,
    totalRepos: 0,
    languages: {}
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadGitHubData = async () => {
      try {
        const githubStats = await fetchGitHubStats();
        setStats(githubStats);
      } catch (error) {
        console.error('Error loading GitHub data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadGitHubData();
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 lg:py-32">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-900/50 via-primary-800/30 to-primary-700/50"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6">
              Revolutionary
              <span className="block gradient-text">Mathematical Library</span>
            </h1>
            <p className="text-xl md:text-2xl text-white/80 mb-8 max-w-3xl mx-auto">
              Balansis implements Absolute Compensation Theory (ACT) for precise numerical computations, 
              eliminating floating-point errors and enabling perfect mathematical operations.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/download"
                className="bg-accent-500 hover:bg-accent-600 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors flex items-center justify-center space-x-2"
              >
                <span>Get Started</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                to="/docs"
                className="border-2 border-white/30 hover:border-white/50 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
              >
                View Documentation
              </Link>
            </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Star className="h-5 w-5 text-yellow-400 mr-2" />
                <span className="text-2xl font-bold text-white">
                  {loading ? '...' : stats.totalStars}
                </span>
              </div>
              <p className="text-white/60">GitHub Stars</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <GitFork className="h-5 w-5 text-blue-400 mr-2" />
                <span className="text-2xl font-bold text-white">
                  {loading ? '...' : stats.totalForks}
                </span>
              </div>
              <p className="text-white/60">Forks</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Download className="h-5 w-5 text-green-400 mr-2" />
                <span className="text-2xl font-bold text-white">
                  {loading ? '...' : stats.totalRepos}
                </span>
              </div>
              <p className="text-white/60">Repositories</p>
            </div>
          </div>
          </div>

          {/* Animated Math Demo */}
          <div className="mb-20">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                See the Difference
              </h2>
              <p className="text-xl text-white/70 max-w-3xl mx-auto">
                Watch how Balansis eliminates floating-point errors that plague traditional mathematics
              </p>
            </div>
            
            <AnimatedMathDemo />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/5 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Why Choose Balansis?
            </h2>
            <p className="text-xl text-white/70 max-w-3xl mx-auto">
              Built on revolutionary Absolute Compensation Theory, Balansis provides 
              unprecedented precision and reliability for mathematical computations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Calculator className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Perfect Precision</h3>
              <p className="text-white/70">
                Eliminate floating-point errors with AbsoluteValue operations that maintain 
                mathematical exactness in all computations.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Infinity className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Eternity Operations</h3>
              <p className="text-white/70">
                Handle infinite values and undefined operations with EternalRatio, 
                providing mathematical completeness.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Compensation Logic</h3>
              <p className="text-white/70">
                Advanced error compensation algorithms ensure stability and accuracy 
                in complex mathematical operations.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Type Safety</h3>
              <p className="text-white/70">
                Full TypeScript support with comprehensive type definitions for 
                safe and reliable mathematical operations.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Code className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Easy Integration</h3>
              <p className="text-white/70">
                Simple API design that integrates seamlessly with existing Python 
                codebases and scientific computing workflows.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 hover:bg-white/15 transition-colors">
              <div className="w-12 h-12 bg-accent-500 rounded-lg flex items-center justify-center mb-4">
                <Users className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Active Community</h3>
              <p className="text-white/70">
                Join a growing community of mathematicians, researchers, and developers 
                pushing the boundaries of computational mathematics.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Experience Perfect Mathematics?
          </h2>
          <p className="text-xl text-white/70 mb-8">
            Join thousands of researchers and developers who trust Balansis for their 
            most critical mathematical computations.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/download"
              className="bg-accent-500 hover:bg-accent-600 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
            >
              Install Balansis
            </Link>
            <Link
              to="/examples"
              className="border-2 border-white/30 hover:border-white/50 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
            >
              Explore Examples
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}