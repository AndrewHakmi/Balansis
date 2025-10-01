import { Github, Twitter, Mail, Heart } from 'lucide-react';

export function Footer() {
  return (
    <footer className="bg-white/5 backdrop-blur-md border-t border-white/10 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-br from-accent-400 to-accent-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">B</span>
              </div>
              <span className="text-white font-bold text-xl">Balansis</span>
            </div>
            <p className="text-white/70 text-sm max-w-md">
              Revolutionary mathematical library implementing Absolute Compensation Theory (ACT) 
              for precise numerical computations and advanced mathematical operations.
            </p>
            <div className="flex items-center space-x-4 mt-6">
              <a
                href="https://github.com/balansis/balansis"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/60 hover:text-white transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://twitter.com/balansis"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/60 hover:text-white transition-colors"
              >
                <Twitter className="w-5 h-5" />
              </a>
              <a
                href="mailto:contact@balansis.org"
                className="text-white/60 hover:text-white transition-colors"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="/docs" className="text-white/60 hover:text-white transition-colors text-sm">
                  Documentation
                </a>
              </li>
              <li>
                <a href="/examples" className="text-white/60 hover:text-white transition-colors text-sm">
                  Examples
                </a>
              </li>
              <li>
                <a href="/theory" className="text-white/60 hover:text-white transition-colors text-sm">
                  ACT Theory
                </a>
              </li>
              <li>
                <a href="/community" className="text-white/60 hover:text-white transition-colors text-sm">
                  Community
                </a>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li>
                <a href="/download" className="text-white/60 hover:text-white transition-colors text-sm">
                  Download
                </a>
              </li>
              <li>
                <a href="https://github.com/balansis/balansis/releases" className="text-white/60 hover:text-white transition-colors text-sm">
                  Releases
                </a>
              </li>
              <li>
                <a href="https://github.com/balansis/balansis/issues" className="text-white/60 hover:text-white transition-colors text-sm">
                  Issues
                </a>
              </li>
              <li>
                <a href="https://github.com/balansis/balansis/blob/main/CONTRIBUTING.md" className="text-white/60 hover:text-white transition-colors text-sm">
                  Contributing
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-white/10 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-white/50 text-sm">
            © 2024 Balansis. All rights reserved.
          </p>
          <p className="text-white/50 text-sm flex items-center mt-4 md:mt-0">
            Made with <Heart className="w-4 h-4 mx-1 text-red-400" /> for mathematics
          </p>
        </div>
      </div>
    </footer>
  );
}