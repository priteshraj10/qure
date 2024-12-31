import Head from 'next/head';
import { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  return (
    <>
      <Head>
        <title>Medical Vision Analysis</title>
        <meta name="description" content="AI-powered medical image analysis" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto py-6 px-4">
            <h1 className="text-3xl font-bold text-gray-900">
              Medical Vision Analysis
            </h1>
          </div>
        </header>

        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {children}
        </main>

        <footer className="bg-white border-t">
          <div className="max-w-7xl mx-auto py-4 px-4 text-center text-gray-600">
            <p>© 2024 Medical Vision Analysis. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </>
  );
}; 