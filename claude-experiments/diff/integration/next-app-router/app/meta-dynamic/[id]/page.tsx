import type { Metadata } from "next";
export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  return { title: `Item ${id}`, description: `Details for item ${id}` };
}
export default async function ItemPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <main><h1 id="item">Item {id}</h1></main>;
}
