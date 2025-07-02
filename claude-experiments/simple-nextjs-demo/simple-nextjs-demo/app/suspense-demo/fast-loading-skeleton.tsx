export default function FastLoadingSkeleton() {
  return (
    <div className="space-y-4">
      <div className="p-4 rounded-lg bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
        <div className="flex items-center justify-between mb-2">
          <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded w-32 animate-pulse"></div>
          <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded-full w-20 animate-pulse"></div>
        </div>
        <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-48 animate-pulse"></div>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="relative overflow-hidden">
            <div className="p-4 bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 rounded-lg">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer"></div>
              <div className="relative">
                <div className="h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2 animate-pulse"></div>
                <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-16 mx-auto animate-pulse"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="flex items-center justify-center space-x-2 mt-4">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
    </div>
  );
}