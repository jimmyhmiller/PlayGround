// Parallel-routes layout: `team` and `analytics` are @slot props the adapter fills by
// matching the URL inside each @slot subtree; `children` is the implicit page.tsx slot.
export default function DashboardLayout({
  children, team, analytics,
}: {
  children: React.ReactNode; team: React.ReactNode; analytics: React.ReactNode;
}) {
  return (
    <section id="dashboard">
      <div id="children-slot">{children}</div>
      <div id="team-slot">{team}</div>
      <div id="analytics-slot">{analytics}</div>
    </section>
  );
}
