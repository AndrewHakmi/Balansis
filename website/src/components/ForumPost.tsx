import { useState } from 'react';
import { MessageCircle, Heart, User, Calendar, Reply } from 'lucide-react';
import { ForumPost as ForumPostType, ForumReply } from '../lib/supabase';
import { useAuth } from '../hooks/useAuth';
import { createForumReply, likeForumPost } from '../lib/supabase';

interface ForumPostProps {
  post: ForumPostType;
  onReply?: (postId: string, reply: ForumReply) => void;
}

export function ForumPost({ post, onReply }: ForumPostProps) {
  const { user, profile } = useAuth();
  const [showReplyForm, setShowReplyForm] = useState(false);
  const [replyContent, setReplyContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLiking, setIsLiking] = useState(false);

  const handleReply = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !replyContent.trim()) return;

    setIsSubmitting(true);
    try {
      const reply = await createForumReply(post.id, replyContent.trim());
      if (reply && onReply) {
        onReply(post.id, reply);
      }
      setReplyContent('');
      setShowReplyForm(false);
    } catch (error) {
      console.error('Error creating reply:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLike = async () => {
    if (!user || isLiking) return;

    setIsLiking(true);
    try {
      await likeForumPost(post.id);
    } catch (error) {
      console.error('Error liking post:', error);
    } finally {
      setIsLiking(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
      {/* Post Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          {post.profiles?.avatar_url ? (
            <img
              src={post.profiles.avatar_url}
              alt={post.profiles.full_name || 'User'}
              className="w-10 h-10 rounded-full"
            />
          ) : (
            <div className="w-10 h-10 rounded-full bg-accent-500 flex items-center justify-center">
              <User className="h-5 w-5 text-white" />
            </div>
          )}
          <div>
            <h3 className="text-white font-semibold">
              {post.profiles?.full_name || 'Anonymous User'}
            </h3>
            <div className="flex items-center space-x-2 text-white/60 text-sm">
              <Calendar className="h-4 w-4" />
              <span>{formatDate(post.created_at)}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4 text-white/60">
          <div className="flex items-center space-x-1">
            <MessageCircle className="h-4 w-4" />
            <span>{post.reply_count || 0}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Heart className="h-4 w-4" />
            <span>{post.like_count || 0}</span>
          </div>
        </div>
      </div>

      {/* Post Content */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-white mb-2">{post.title}</h2>
        <p className="text-white/80 leading-relaxed">{post.content}</p>
      </div>

      {/* Post Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-white/20">
        <div className="flex items-center space-x-4">
          <button
            onClick={handleLike}
            disabled={!user || isLiking}
            className="flex items-center space-x-2 text-white/60 hover:text-accent-400 transition-colors disabled:opacity-50"
          >
            <Heart className="h-4 w-4" />
            <span>Like</span>
          </button>
          <button
            onClick={() => setShowReplyForm(!showReplyForm)}
            disabled={!user}
            className="flex items-center space-x-2 text-white/60 hover:text-accent-400 transition-colors disabled:opacity-50"
          >
            <Reply className="h-4 w-4" />
            <span>Reply</span>
          </button>
        </div>
        {!user && (
          <p className="text-white/40 text-sm">Sign in to interact</p>
        )}
      </div>

      {/* Reply Form */}
      {showReplyForm && user && (
        <form onSubmit={handleReply} className="mt-4 pt-4 border-t border-white/20">
          <div className="flex items-start space-x-3">
            {profile?.avatar_url ? (
              <img
                src={profile.avatar_url}
                alt={profile.full_name || 'User'}
                className="w-8 h-8 rounded-full"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-accent-500 flex items-center justify-center">
                <User className="h-4 w-4 text-white" />
              </div>
            )}
            <div className="flex-1">
              <textarea
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                placeholder="Write your reply..."
                className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-accent-500 resize-none"
                rows={3}
                required
              />
              <div className="flex items-center justify-end space-x-2 mt-2">
                <button
                  type="button"
                  onClick={() => {
                    setShowReplyForm(false);
                    setReplyContent('');
                  }}
                  className="px-4 py-2 text-white/60 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting || !replyContent.trim()}
                  className="bg-accent-500 hover:bg-accent-600 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
                >
                  {isSubmitting ? 'Posting...' : 'Post Reply'}
                </button>
              </div>
            </div>
          </div>
        </form>
      )}

      {/* Replies */}
      {post.forum_replies && post.forum_replies.length > 0 && (
        <div className="mt-6 pt-4 border-t border-white/20 space-y-4">
          <h4 className="text-white font-medium">Replies</h4>
          {post.forum_replies.map((reply) => (
            <div key={reply.id} className="flex items-start space-x-3 bg-white/5 rounded-lg p-4">
              {reply.profiles?.avatar_url ? (
                <img
                  src={reply.profiles.avatar_url}
                  alt={reply.profiles.full_name || 'User'}
                  className="w-8 h-8 rounded-full"
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-accent-500 flex items-center justify-center">
                  <User className="h-4 w-4 text-white" />
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <span className="text-white font-medium text-sm">
                    {reply.profiles?.full_name || 'Anonymous User'}
                  </span>
                  <span className="text-white/40 text-xs">
                    {formatDate(reply.created_at)}
                  </span>
                </div>
                <p className="text-white/80 text-sm">{reply.content}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}