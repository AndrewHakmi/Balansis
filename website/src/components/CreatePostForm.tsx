import { useState } from 'react';
import { Plus, User } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { createForumPost } from '../lib/supabase';
import { ForumPost } from '../lib/supabase';

interface CreatePostFormProps {
  onPostCreated?: (post: ForumPost) => void;
}

export function CreatePostForm({ onPostCreated }: CreatePostFormProps) {
  const { user, profile } = useAuth();
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user || !title.trim() || !content.trim()) return;

    setIsSubmitting(true);
    try {
      const post = await createForumPost(title.trim(), content.trim());
      if (post && onPostCreated) {
        onPostCreated(post);
      }
      setTitle('');
      setContent('');
      setShowForm(false);
    } catch (error) {
      console.error('Error creating post:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!user) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 text-center">
        <p className="text-white/70 mb-4">Sign in to create a new discussion post</p>
        <p className="text-white/50 text-sm">Join the community to share your thoughts and questions about Balansis</p>
      </div>
    );
  }

  if (!showForm) {
    return (
      <button
        onClick={() => setShowForm(true)}
        className="w-full bg-accent-500 hover:bg-accent-600 text-white p-6 rounded-xl font-medium transition-colors flex items-center justify-center space-x-2 border border-accent-400"
      >
        <Plus className="h-5 w-5" />
        <span>Start a New Discussion</span>
      </button>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
      <div className="flex items-center space-x-3 mb-6">
        {profile?.avatar_url ? (
          <img
            src={profile.avatar_url}
            alt={profile.full_name || 'User'}
            className="w-10 h-10 rounded-full"
          />
        ) : (
          <div className="w-10 h-10 rounded-full bg-accent-500 flex items-center justify-center">
            <User className="h-5 w-5 text-white" />
          </div>
        )}
        <div>
          <h3 className="text-white font-semibold">Create New Discussion</h3>
          <p className="text-white/60 text-sm">Share your thoughts with the community</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="title" className="block text-white font-medium mb-2">
            Title
          </label>
          <input
            type="text"
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="What would you like to discuss?"
            className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-accent-500"
            required
            maxLength={200}
          />
          <p className="text-white/40 text-sm mt-1">{title.length}/200 characters</p>
        </div>

        <div>
          <label htmlFor="content" className="block text-white font-medium mb-2">
            Content
          </label>
          <textarea
            id="content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Share your thoughts, questions, or ideas about Balansis..."
            className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-accent-500 resize-none"
            rows={6}
            required
            maxLength={2000}
          />
          <p className="text-white/40 text-sm mt-1">{content.length}/2000 characters</p>
        </div>

        <div className="flex items-center justify-end space-x-3 pt-4">
          <button
            type="button"
            onClick={() => {
              setShowForm(false);
              setTitle('');
              setContent('');
            }}
            className="px-6 py-2 text-white/60 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting || !title.trim() || !content.trim()}
            className="bg-accent-500 hover:bg-accent-600 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center space-x-2"
          >
            {isSubmitting ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Creating...</span>
              </>
            ) : (
              <>
                <Plus className="h-4 w-4" />
                <span>Create Post</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}