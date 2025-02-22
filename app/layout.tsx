import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CS 4641 Group 58',
  description: 'ML Project Group 58',
  generator: 'ML Project Group 58',
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
