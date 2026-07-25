// Hosts the @modal parallel slot. On a hard load the slot renders its default (null);
// the intercept overlay is portaled by the client Router on soft-nav.
export default function GalleryLayout({
  children, modal,
}: {
  children: React.ReactNode; modal: React.ReactNode;
}) {
  return (<div id="gallery-root">{children}{modal}</div>);
}
