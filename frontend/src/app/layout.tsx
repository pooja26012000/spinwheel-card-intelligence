import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SpinWheel Card Intelligence',
  description: 'AI-powered sports card grading and valuation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
