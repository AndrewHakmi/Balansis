import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://xfblavnbqnfwjvfggjel.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhmYmxhdm5icW5md2p2ZmdnamVsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkzNTczMDcsImV4cCI6MjA3NDkzMzMwN30.jB_HP7pFvuQzheseNdRUff-gOgU6hhBhscyNX4jpWsk';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  }
});

// Database types
export interface Profile {
  id: string;
  username: string;
  full_name?: string;
  avatar_url?: string;
  website?: string;
  github_username?: string;
  created_at: string;
  updated_at: string;
}

export interface ForumPost {
  id: string;
  title: string;
  content: string;
  user_id: string;
  like_count?: number;
  reply_count?: number;
  created_at: string;
  updated_at?: string;
  profiles?: Profile;
  forum_replies?: ForumReply[];
}

export interface ForumReply {
  id: string;
  post_id: string;
  content: string;
  user_id: string;
  created_at: string;
  updated_at?: string;
  profiles?: Profile;
}

// Auth helpers
export const signInWithGitHub = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: `${window.location.origin}/auth/callback`
    }
  });
  return { data, error };
};

export const signOut = async () => {
  const { error } = await supabase.auth.signOut();
  return { error };
};

export const getCurrentUser = async () => {
  const { data: { user }, error } = await supabase.auth.getUser();
  return { user, error };
};

// Profile helpers
export const getProfile = async (userId: string) => {
  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single();
  return { data, error };
};

export const updateProfile = async (userId: string, updates: Partial<Profile>) => {
  const { data, error } = await supabase
    .from('profiles')
    .update(updates)
    .eq('id', userId)
    .select()
    .single();
  return { data, error };
};

// Forum functions
export async function getForumPosts(): Promise<ForumPost[]> {
  const { data, error } = await supabase
    .from('forum_posts')
    .select(`
      *,
      profiles (
        id,
        full_name,
        avatar_url
      ),
      forum_replies (
        id,
        content,
        created_at,
        profiles (
          id,
          full_name,
          avatar_url
        )
      )
    `)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data || [];
}

export async function createForumPost(title: string, content: string): Promise<ForumPost | null> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) throw new Error('User not authenticated');

  const { data, error } = await supabase
    .from('forum_posts')
    .insert({
      title,
      content,
      user_id: user.id
    })
    .select(`
      *,
      profiles (
        id,
        full_name,
        avatar_url
      )
    `)
    .single();

  if (error) throw error;
  return data;
}

export async function createForumReply(postId: string, content: string): Promise<ForumReply | null> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) throw new Error('User not authenticated');

  const { data, error } = await supabase
    .from('forum_replies')
    .insert({
      post_id: postId,
      content,
      user_id: user.id
    })
    .select(`
      *,
      profiles (
        id,
        full_name,
        avatar_url
      )
    `)
    .single();

  if (error) throw error;
  return data;
}

export async function likeForumPost(postId: string): Promise<void> {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) throw new Error('User not authenticated');

  // Check if user already liked this post
  const { data: existingLike } = await supabase
    .from('forum_likes')
    .select('id')
    .eq('post_id', postId)
    .eq('user_id', user.id)
    .single();

  if (existingLike) {
    // Unlike the post
    const { error } = await supabase
      .from('forum_likes')
      .delete()
      .eq('post_id', postId)
      .eq('user_id', user.id);
    
    if (error) throw error;
  } else {
    // Like the post
    const { error } = await supabase
      .from('forum_likes')
      .insert({
        post_id: postId,
        user_id: user.id
      });
    
    if (error) throw error;
  }
}