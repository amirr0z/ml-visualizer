
import type { Metadata } from "next";
import ThemeProvider from "@/components/providers/ThemeProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML Laboratory",
  description:
    "An interactive laboratory for learning machine learning algorithms.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
