async function fetchFastData() {
  // Randomly fast (100ms) or slow (500ms) to demonstrate the delayed loading
  const delay = Math.random() > 0.5 ? 100 : 500;
  await new Promise(resolve => setTimeout(resolve, delay));
  
  return {
    responseTime: delay,
    data: {
      status: delay < 200 ? 'Fast' : 'Slow',
      message: `Response took ${delay}ms`,
      color: delay < 200 ? 'green' : 'yellow',
      items: [
        { id: 1, value: Math.floor(Math.random() * 100) },
        { id: 2, value: Math.floor(Math.random() * 100) },
        { id: 3, value: Math.floor(Math.random() * 100) }
      ]
    }
  };
}

export default async function FastServerComponent() {
  const result = await fetchFastData();
  
  return (
    <div className="space-y-4">
      <div className={`p-4 rounded-lg ${
        result.data.color === 'green' 
          ? 'bg-green-50 dark:bg-green-900/20 border-2 border-green-500' 
          : 'bg-yellow-50 dark:bg-yellow-900/20 border-2 border-yellow-500'
      }`}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-lg">Response Status</h3>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            result.data.color === 'green'
              ? 'bg-green-500 text-white'
              : 'bg-yellow-500 text-black'
          }`}>
            {result.data.status}
          </span>
        </div>
        <p className="text-gray-600 dark:text-gray-400">{result.data.message}</p>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        {result.data.items.map(item => (
          <div key={item.id} className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg text-center">
            <p className="text-2xl font-bold">{item.value}</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">Value {item.id}</p>
          </div>
        ))}
      </div>
      
      <div className="text-sm text-gray-500 dark:text-gray-400 text-center">
        Refresh the page to see different response times.
        <br />
        Loading indicator only shows if response takes &gt; 200ms
      </div>
    </div>
  );
}