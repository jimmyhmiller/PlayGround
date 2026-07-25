export default async function PhotoModal({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return (
    <div id="photo-modal" style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)" }}>
      <p id="modal-content">Photo {id} in a modal</p>
    </div>
  );
}
