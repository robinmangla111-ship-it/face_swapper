import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Face Swapper — AI-Powered Face Replacement",
  description:
    "Swap faces in videos with AI. Upload a video, select a character, provide a source face, and get your result in minutes.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-zinc-950">
        <div className="min-h-screen flex flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
