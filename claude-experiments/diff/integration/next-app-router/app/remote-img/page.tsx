import Image from "next/image";
export const dynamic = "force-dynamic";
export default function RemoteImgPage() {
  return (
    <main>
      <h1 id="remote-heading">Remote image</h1>
      <Image src="https://assets.imgix.net/example.png" alt="remote" width={200} height={100} />
    </main>
  );
}
