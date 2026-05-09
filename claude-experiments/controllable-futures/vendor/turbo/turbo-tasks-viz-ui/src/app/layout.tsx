import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'turbo-tasks Visualizer',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  )
}
