'use client';

import { useEffect, useState } from 'react';
import LoadingSkeleton from './loading-skeleton';

interface UserData {
  user: {
    id: number;
    name: string;
    email: string;
    avatar: string;
  };
  posts: Array<{
    id: number;
    title: string;
    content: string;
  }>;
  stats: {
    totalPosts: number;
    totalLikes: number;
    joinedDate: string;
  };
}

export default function ClientFetchComponent() {
  const [data, setData] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        // Simulate API call with 2 second delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Simulate fetching data
        const userData: UserData = {
          user: {
            id: 2,
            name: "Jane Smith",
            email: "jane@example.com",
            avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Jane"
          },
          posts: [
            { id: 1, title: "Hello World", content: "My first blog post here" },
            { id: 2, title: "React Tips", content: "Some useful React patterns" },
            { id: 3, title: "Web Performance", content: "How to optimize your app" }
          ],
          stats: {
            totalPosts: 3,
            totalLikes: 67,
            joinedDate: "2024-02-20"
          }
        };
        
        setData(userData);
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  if (loading) {
    return <LoadingSkeleton />;
  }

  if (error || !data) {
    return (
      <div className="text-red-500 p-4 text-center">
        {error || 'Something went wrong'}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-orange-600 rounded-full flex items-center justify-center text-white font-bold text-xl">
          {data.user.name.charAt(0)}
        </div>
        <div>
          <h3 className="font-semibold text-lg">{data.user.name}</h3>
          <p className="text-gray-600 dark:text-gray-400">{data.user.email}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
          <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {data.stats.totalPosts}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Posts</p>
        </div>
        <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded">
          <p className="text-2xl font-bold text-pink-600 dark:text-pink-400">
            {data.stats.totalLikes}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Likes</p>
        </div>
        <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded">
          <p className="text-lg font-bold text-indigo-600 dark:text-indigo-400">
            Feb 2024
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Joined</p>
        </div>
      </div>
      
      <div className="space-y-2">
        <h4 className="font-semibold">Recent Posts</h4>
        {data.posts.map(post => (
          <div key={post.id} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
            <h5 className="font-medium">{post.title}</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">{post.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}