import { useState, useEffect } from 'react';
import { Users, MessageSquare, MessageCircle, BookOpen, Github, Calendar, Plus, Award } from 'lucide-react';
import { ForumPost as ForumPostComponent } from '../components/ForumPost';
import { CreatePostForm } from '../components/CreatePostForm';
import { getForumPosts, ForumPost, ForumReply } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';

export function Community() {
  const { user } = useAuth();
  const [posts, setPosts] = useState<ForumPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'recent' | 'popular'>('recent');

  useEffect(() => {
    loadPosts();
  }, []);

  const loadPosts = async () => {
    try {
      setLoading(true);
      const forumPosts = await getForumPosts();
      setPosts(forumPosts);
    } catch (error) {
      console.error('Error loading posts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePostCreated = (newPost: ForumPost) => {
    setPosts([newPost, ...posts]);
  };

  const handleReply = (postId: string, reply: ForumReply) => {
    setPosts(posts.map(post => 
      post.id === postId 
        ? { 
            ...post, 
            reply_count: (post.reply_count || 0) + 1,
            forum_replies: [...(post.forum_replies || []), reply]
          }
        : post
    ));
  };

  return (
    <div className="min-h-screen py-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Community
          </h1>
          <p className="text-xl text-white/70 max-w-3xl mx-auto">
            Join a vibrant community of mathematicians, researchers, and developers 
            advancing the frontiers of computational mathematics with ACT.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          {/* Community Stats */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
              <h3 className="text-xl font-semibold text-white mb-4">Community Stats</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-white/70">Active Members</span>
                  <span className="text-accent-400 font-bold">2,847</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-white/70">GitHub Stars</span>
                  <span className="text-accent-400 font-bold">1,234</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-white/70">Contributors</span>
                  <span className="text-accent-400 font-bold">89</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-white/70">Research Papers</span>
                  <span className="text-accent-400 font-bold">23</span>
                </div>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4">Quick Links</h3>
              <div className="space-y-3">
                <a href="#" className="flex items-center space-x-3 text-white/70 hover:text-white transition-colors">
                  <Github className="w-5 h-5" />
                  <span>GitHub Repository</span>
                </a>
                <a href="#" className="flex items-center space-x-3 text-white/70 hover:text-white transition-colors">
                  <MessageCircle className="w-5 h-5" />
                  <span>Discord Server</span>
                </a>
                <a href="#" className="flex items-center space-x-3 text-white/70 hover:text-white transition-colors">
                  <BookOpen className="w-5 h-5" />
                  <span>Research Forum</span>
                </a>
                <a href="#" className="flex items-center space-x-3 text-white/70 hover:text-white transition-colors">
                  <Calendar className="w-5 h-5" />
                  <span>Events Calendar</span>
                </a>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Discussion Forum */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-semibold text-white flex items-center">
                  <MessageSquare className="w-6 h-6 mr-3 text-accent-400" />
                  Community Forum
                </h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setActiveTab('recent')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeTab === 'recent'
                        ? 'bg-accent-500 text-white'
                        : 'text-white/60 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    Recent
                  </button>
                  <button
                    onClick={() => setActiveTab('popular')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeTab === 'popular'
                        ? 'bg-accent-500 text-white'
                        : 'text-white/60 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    Popular
                  </button>
                </div>
              </div>

              <div className="space-y-6">
                <CreatePostForm onPostCreated={handlePostCreated} />
                
                {loading ? (
                  <div className="space-y-6">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="bg-white/5 rounded-lg p-4 animate-pulse">
                        <div className="flex items-start space-x-3 mb-4">
                          <div className="w-10 h-10 rounded-full bg-white/20"></div>
                          <div className="flex-1">
                            <div className="h-4 bg-white/20 rounded w-1/4 mb-2"></div>
                            <div className="h-3 bg-white/20 rounded w-1/6"></div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="h-5 bg-white/20 rounded w-3/4"></div>
                          <div className="h-4 bg-white/20 rounded w-full"></div>
                          <div className="h-4 bg-white/20 rounded w-2/3"></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : posts.length > 0 ? (
                  <div className="space-y-6">
                    {posts
                      .sort((a, b) => {
                        if (activeTab === 'popular') {
                          return (b.like_count || 0) - (a.like_count || 0);
                        }
                        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
                      })
                      .map((post) => (
                        <ForumPostComponent
                          key={post.id}
                          post={post}
                          onReply={handleReply}
                        />
                      ))}
                  </div>
                ) : (
                  <div className="bg-white/5 rounded-lg p-12 text-center">
                    <MessageSquare className="h-16 w-16 text-white/40 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">No discussions yet</h3>
                    <p className="text-white/60 mb-6">
                      Be the first to start a conversation about Balansis!
                    </p>
                    {user && (
                      <button
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="bg-accent-500 hover:bg-accent-600 text-white px-6 py-3 rounded-lg font-medium transition-colors inline-flex items-center space-x-2"
                      >
                        <Plus className="h-4 w-4" />
                        <span>Start Discussion</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Contributors */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-2xl font-semibold text-white mb-6 flex items-center">
                <Users className="w-6 h-6 mr-3 text-accent-400" />
                Top Contributors
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-lg p-4 flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-accent-400 to-accent-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">DR</span>
                  </div>
                  <div>
                    <h4 className="text-white font-medium">Dr. Research</h4>
                    <p className="text-white/70 text-sm">Core Algorithm Developer</p>
                    <p className="text-accent-400 text-xs">247 commits</p>
                  </div>
                </div>

                <div className="bg-white/5 rounded-lg p-4 flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary-400 to-primary-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">MQ</span>
                  </div>
                  <div>
                    <h4 className="text-white font-medium">MathQueen</h4>
                    <p className="text-white/70 text-sm">Theory Specialist</p>
                    <p className="text-accent-400 text-xs">189 commits</p>
                  </div>
                </div>

                <div className="bg-white/5 rounded-lg p-4 flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">CP</span>
                  </div>
                  <div>
                    <h4 className="text-white font-medium">CodePython</h4>
                    <p className="text-white/70 text-sm">Implementation Expert</p>
                    <p className="text-accent-400 text-xs">156 commits</p>
                  </div>
                </div>

                <div className="bg-white/5 rounded-lg p-4 flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-purple-600 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold">TS</span>
                  </div>
                  <div>
                    <h4 className="text-white font-medium">TestSuite</h4>
                    <p className="text-white/70 text-sm">Quality Assurance</p>
                    <p className="text-accent-400 text-xs">134 commits</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Events & News */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
            <h3 className="text-2xl font-semibold text-white mb-6 flex items-center">
              <Calendar className="w-6 h-6 mr-3 text-accent-400" />
              Upcoming Events
            </h3>
            <div className="space-y-4">
              <div className="border-l-4 border-accent-400 pl-4">
                <h4 className="text-white font-medium">ACT Workshop 2024</h4>
                <p className="text-white/70 text-sm">Deep dive into advanced compensation techniques</p>
                <p className="text-accent-400 text-xs">March 15, 2024 • Online</p>
              </div>
              <div className="border-l-4 border-primary-400 pl-4">
                <h4 className="text-white font-medium">Community Meetup</h4>
                <p className="text-white/70 text-sm">Monthly virtual gathering for all members</p>
                <p className="text-accent-400 text-xs">March 22, 2024 • Discord</p>
              </div>
              <div className="border-l-4 border-green-400 pl-4">
                <h4 className="text-white font-medium">Research Symposium</h4>
                <p className="text-white/70 text-sm">Latest findings in computational mathematics</p>
                <p className="text-accent-400 text-xs">April 5, 2024 • MIT</p>
              </div>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
            <h3 className="text-2xl font-semibold text-white mb-6 flex items-center">
              <Award className="w-6 h-6 mr-3 text-accent-400" />
              Recognition
            </h3>
            <div className="space-y-4">
              <div className="bg-white/5 rounded-lg p-4">
                <h4 className="text-white font-medium mb-2">Contributor of the Month</h4>
                <p className="text-white/70 text-sm mb-2">
                  @alice_researcher for outstanding work on parallel compensation algorithms
                </p>
                <span className="text-accent-400 text-xs">February 2024</span>
              </div>
              <div className="bg-white/5 rounded-lg p-4">
                <h4 className="text-white font-medium mb-2">Best Research Paper</h4>
                <p className="text-white/70 text-sm mb-2">
                  "Quantum-Resistant ACT Implementation" by Dr. Quantum
                </p>
                <span className="text-accent-400 text-xs">ACT Conference 2023</span>
              </div>
            </div>
          </div>
        </div>

        {/* Get Involved */}
        <div className="bg-gradient-to-r from-accent-500/20 to-primary-500/20 backdrop-blur-md rounded-xl p-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">Get Involved</h2>
          <p className="text-white/70 mb-8 max-w-2xl mx-auto">
            Whether you're a mathematician, researcher, developer, or enthusiast, 
            there are many ways to contribute to the Balansis community.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button className="bg-accent-500 hover:bg-accent-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              Join Discord
            </button>
            <button className="bg-primary-600 hover:bg-primary-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              Contribute Code
            </button>
            <button className="border-2 border-white/30 hover:border-white/50 text-white px-6 py-3 rounded-lg font-semibold transition-colors">
              Share Research
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}