import ClientComponent from './client-component';
import Link from 'next/link';

async function getData() {
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  return {
    message: "Hello from the server!",
    timestamp: new Date().toISOString(),
    items: [
      { id: 1, name: "Item 1" },
      { id: 2, name: "Item 2" },
      { id: 3, name: "Item 3" }
    ]
  };
}

export default async function Home() {
  const data = await getData();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold mb-8">Server Component Demo</h1>
      <ClientComponent data={data} />
      <Link 
        href="/suspense-demo" 
        className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded transition-colors"
      >
        View Suspense vs Client Fetching Demo
      </Link>
    </main>
  );
}