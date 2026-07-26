export const metadata = { title: "Scale", description: "3000-page scale benchmark" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
