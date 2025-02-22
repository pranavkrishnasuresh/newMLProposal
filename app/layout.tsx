import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CS 4641 Group 58',
  description: 'Krishna',
  generator: 'Krishna',
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
