// GitHub API integration for repository statistics
export interface GitHubRepo {
  name: string;
  full_name: string;
  description: string;
  html_url: string;
  stargazers_count: number;
  forks_count: number;
  open_issues_count: number;
  language: string;
  updated_at: string;
  topics: string[];
}

export interface GitHubStats {
  totalStars: number;
  totalForks: number;
  totalRepos: number;
  languages: { [key: string]: number };
}

const GITHUB_API_BASE = 'https://api.github.com';
const GITHUB_USERNAME = 'balansis'; // Replace with actual GitHub username/organization

export async function fetchGitHubRepos(): Promise<GitHubRepo[]> {
  try {
    const response = await fetch(`${GITHUB_API_BASE}/users/${GITHUB_USERNAME}/repos?sort=updated&per_page=10`);
    
    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }
    
    const repos = await response.json();
    return repos.map((repo: any) => ({
      name: repo.name,
      full_name: repo.full_name,
      description: repo.description || 'No description available',
      html_url: repo.html_url,
      stargazers_count: repo.stargazers_count,
      forks_count: repo.forks_count,
      open_issues_count: repo.open_issues_count,
      language: repo.language || 'Unknown',
      updated_at: repo.updated_at,
      topics: repo.topics || []
    }));
  } catch (error) {
    console.error('Error fetching GitHub repos:', error);
    return [];
  }
}

export async function fetchGitHubStats(): Promise<GitHubStats> {
  try {
    const repos = await fetchGitHubRepos();
    
    const stats: GitHubStats = {
      totalStars: 0,
      totalForks: 0,
      totalRepos: repos.length,
      languages: {}
    };
    
    repos.forEach(repo => {
      stats.totalStars += repo.stargazers_count;
      stats.totalForks += repo.forks_count;
      
      if (repo.language) {
        stats.languages[repo.language] = (stats.languages[repo.language] || 0) + 1;
      }
    });
    
    return stats;
  } catch (error) {
    console.error('Error calculating GitHub stats:', error);
    return {
      totalStars: 0,
      totalForks: 0,
      totalRepos: 0,
      languages: {}
    };
  }
}

export async function fetchLatestRelease(): Promise<any> {
  try {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${GITHUB_USERNAME}/balansis/releases/latest`);
    
    if (!response.ok) {
      // If no releases found, return mock data
      return {
        tag_name: 'v1.0.0',
        name: 'Initial Release',
        published_at: new Date().toISOString(),
        html_url: `https://github.com/${GITHUB_USERNAME}/balansis/releases`,
        body: 'Initial release of Balansis mathematical framework.'
      };
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching latest release:', error);
    return {
      tag_name: 'v1.0.0',
      name: 'Initial Release',
      published_at: new Date().toISOString(),
      html_url: `https://github.com/${GITHUB_USERNAME}/balansis/releases`,
      body: 'Initial release of Balansis mathematical framework.'
    };
  }
}