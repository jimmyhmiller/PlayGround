import type { NextPageContext } from "next";
// getInitialProps page: runs per request; on the server it sees ctx.req and echoes
// the request User-Agent, proving the legacy data-fetching lifecycle threads props.
function Gip({ ua }: { ua: string }) {
  return (<main><h1 id="gip">GIP</h1><p id="ua">{ua}</p></main>);
}
Gip.getInitialProps = async (ctx: NextPageContext) => {
  const ua = (ctx.req && (ctx.req.headers["user-agent"] as string)) || "client-nav";
  return { ua };
};
export default Gip;
