// The root Server Component. It has NO directive, so in the react-server graph it
// runs on the server: it `await`s a trivial data fetch, then embeds the `"use
// client"` island and passes the `"use server"` `increment` action into it as a
// prop. In the flight stream the island becomes a client reference and the action
// becomes a server reference — neither's code ships from here. Only the
// react-server render graph imports this module; the SSR and browser graphs
// reconstruct its output from the flight.
import { Counter } from "./Counter";
import { increment } from "./actions";

export async function Page() {
  const data = await Promise.resolve("from-server");
  return (
    <div id="page">
      <h1 id="heading">Server:{data}</h1>
      <Counter initial={5} increment={increment} />
    </div>
  );
}
