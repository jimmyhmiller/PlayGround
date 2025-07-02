import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useCart } from '@/contexts/CartContext';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const router = useRouter();
  const { itemCount } = useCart();

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-md">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="text-2xl font-bold text-amber-800">
              BrewMaster Supplies
            </Link>
            <div className="flex space-x-8">
              <Link 
                href="/" 
                className={`hover:text-amber-700 transition-colors ${
                  router.pathname === '/' ? 'text-amber-700' : 'text-gray-700'
                }`}
              >
                Home
              </Link>
              <Link 
                href="/products" 
                className={`hover:text-amber-700 transition-colors ${
                  router.pathname === '/products' ? 'text-amber-700' : 'text-gray-700'
                }`}
              >
                Products
              </Link>
              <Link 
                href="/cart" 
                className={`hover:text-amber-700 transition-colors relative ${
                  router.pathname === '/cart' ? 'text-amber-700' : 'text-gray-700'
                }`}
              >
                Cart
                {itemCount > 0 && (
                  <span className="absolute -top-2 -right-2 bg-amber-700 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {itemCount}
                  </span>
                )}
              </Link>
            </div>
          </div>
        </div>
      </nav>
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; 2024 BrewMaster Supplies. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;