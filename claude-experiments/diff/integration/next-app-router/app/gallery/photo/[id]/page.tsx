export default async function PhotoPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <main><h1 id="photo-full">Full photo {id}</h1></main>;
}
