import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Video Sentiment Based Thumbnail',
  description: 'Video Sentiment Based Thumbnail',
  generator: 'Anurag',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
