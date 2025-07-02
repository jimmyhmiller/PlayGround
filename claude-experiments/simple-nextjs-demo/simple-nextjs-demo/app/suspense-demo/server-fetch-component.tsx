async function fetchUserData() {
  // Simulate API call with 2 second delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  return {
    user: {
      id: 1,
      name: "John Doe",
      email: "john@example.com",
      avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=John"
    },
    posts: [
      { id: 1, title: "First Post", content: "This is my first post content" },
      { id: 2, title: "Second Post", content: "Another interesting post" },
      { id: 3, title: "Third Post", content: "More content to read" }
    ],
    stats: {
      totalPosts: 3,
      totalLikes: 42,
      joinedDate: "2024-01-15"
    }
  };
}

export default async function ServerFetchComponent() {
  const data = await fetchUserData();
  
  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xl">
          {data.user.name.charAt(0)}
        </div>
        <div>
          <h3 className="font-semibold text-lg">{data.user.name}</h3>
          <p className="text-gray-600 dark:text-gray-400">{data.user.email}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {data.stats.totalPosts}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Posts</p>
        </div>
        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {data.stats.totalLikes}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Likes</p>
        </div>
        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
          <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
            Jan 2024
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