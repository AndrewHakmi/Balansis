import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Github, User, LogOut } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { signInWithGitHub, signOut } from '../lib/supabase';

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const location = useLocation();
  const { user, profile, loading } = useAuth();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Docs', href: '/docs' },
    { name: 'Examples', href: '/examples' },
    { name: 'Theory', href: '/theory' },
    { name: 'Community', href: '/community' },
  ];

  const isActive = (path: string) => location.pathname === path;

  const handleSignIn = async () => {
    try {
      await signInWithGitHub();
    } catch (error) {
      console.error('Sign in error:', error);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      setShowUserMenu(false);
    } catch (error) {
      console.error('Sign out error:', error);
    }
  };

  return (
    <nav className="bg-white/10 backdrop-blur-md border-b border-white/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-accent-400 to-accent-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">B</span>
              </div>
              <span className="text-white font-bold text-xl">Balansis</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? 'text-accent-400 bg-white/10'
                    : 'text-white/80 hover:text-white hover:bg-white/5'
                }`}
              >
                {item.name}
              </Link>
            ))}
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/balansis/balansis"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/70 hover:text-white transition-colors"
              >
                <Github className="h-5 w-5" />
              </a>
              
              {loading ? (
                <div className="w-8 h-8 rounded-full bg-white/20 animate-pulse"></div>
              ) : user ? (
                <div className="relative">
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center space-x-2 text-white/70 hover:text-white transition-colors"
                  >
                    {profile?.avatar_url ? (
                      <img
                        src={profile.avatar_url}
                        alt={profile.full_name || 'User'}
                        className="w-8 h-8 rounded-full"
                      />
                    ) : (
                      <User className="h-5 w-5" />
                    )}
                    <span className="hidden md:block">{profile?.full_name || user.email}</span>
                  </button>
                  
                  {showUserMenu && (
                    <div className="absolute right-0 mt-2 w-48 bg-gray-900 rounded-lg shadow-lg border border-white/20 py-2">
                      <div className="px-4 py-2 border-b border-white/20">
                        <p className="text-white font-medium">{profile?.full_name || 'User'}</p>
                        <p className="text-white/70 text-sm">{user.email}</p>
                      </div>
                      <button
                        onClick={handleSignOut}
                        className="w-full text-left px-4 py-2 text-white/70 hover:text-white hover:bg-white/10 transition-colors flex items-center space-x-2"
                      >
                        <LogOut className="h-4 w-4" />
                        <span>Sign Out</span>
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <button
                  onClick={handleSignIn}
                  className="bg-accent-500 hover:bg-accent-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
                >
                  <Github className="h-4 w-4" />
                  <span>Sign In</span>
                </button>
              )}
              
              <Link
                to="/download"
                className="bg-accent-500 hover:bg-accent-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Download
              </Link>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-white/80 hover:text-white transition-colors"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isOpen && (
        <div className="md:hidden bg-white/10 backdrop-blur-md border-t border-white/20">
          <div className="px-2 pt-2 pb-3 space-y-1">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => setIsOpen(false)}
                className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                  isActive(item.href)
                    ? 'text-accent-400 bg-white/10'
                    : 'text-white/80 hover:text-white hover:bg-white/5'
                }`}
              >
                {item.name}
              </Link>
            ))}
            <div className="border-t border-white/20 pt-3 mt-3 space-y-2">
              <a
                href="https://github.com/balansis/balansis"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center px-3 py-2 text-white/70 hover:text-white transition-colors"
              >
                <Github className="h-5 w-5 mr-2" />
                GitHub
              </a>
              
              {loading ? (
                <div className="px-3 py-2">
                  <div className="w-full h-8 rounded bg-white/20 animate-pulse"></div>
                </div>
              ) : user ? (
                <div className="px-3 py-2 space-y-2">
                  <div className="flex items-center space-x-2 text-white">
                    {profile?.avatar_url ? (
                      <img
                        src={profile.avatar_url}
                        alt={profile.full_name || 'User'}
                        className="w-6 h-6 rounded-full"
                      />
                    ) : (
                      <User className="h-5 w-5" />
                    )}
                    <span className="text-sm">{profile?.full_name || user.email}</span>
                  </div>
                  <button
                    onClick={() => {
                      handleSignOut();
                      setIsOpen(false);
                    }}
                    className="w-full flex items-center space-x-2 px-2 py-1 text-white/70 hover:text-white hover:bg-white/10 rounded transition-colors"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => {
                    handleSignIn();
                    setIsOpen(false);
                  }}
                  className="w-full mx-3 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
                >
                  <Github className="h-4 w-4" />
                  <span>Sign In</span>
                </button>
              )}
              
              <Link
                to="/download"
                className="block mx-3 bg-accent-500 hover:bg-accent-600 text-white px-4 py-2 rounded-lg font-medium transition-colors text-center"
                onClick={() => setIsOpen(false)}
              >
                Download
              </Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}