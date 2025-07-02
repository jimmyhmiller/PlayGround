export default function LoadingSkeleton() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="flex items-center space-x-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="w-16 h-16 bg-gray-300 dark:bg-gray-700 rounded-full"></div>
        <div className="space-y-2 flex-1">
          <div className="h-5 bg-gray-300 dark:bg-gray-700 rounded w-32"></div>
          <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-48"></div>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
          <div className="h-8 bg-gray-300 dark:bg-gray-700 rounded mb-1"></div>
          <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-12 mx-auto"></div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
          <div className="h-8 bg-gray-300 dark:bg-gray-700 rounded mb-1"></div>
          <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-12 mx-auto"></div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded mb-1"></div>
          <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-12 mx-auto"></div>
        </div>
      </div>
      
      <div className="space-y-2">
        <div className="h-5 bg-gray-300 dark:bg-gray-700 rounded w-24 mb-3"></div>
        {[1, 2, 3].map(i => (
          <div key={i} className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
            <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-32 mb-2"></div>
            <div className="h-3 bg-gray-300 dark:bg-gray-700 rounded w-full"></div>
          </div>
        ))}
      </div>
    </div>
  );
}