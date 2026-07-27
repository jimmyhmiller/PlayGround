// The context `next/head` writes into. On the server the render entry provides a
// collector `{ push }`; the custom `_document` reads the collected elements out of
// `DocumentContext`. On the client the entry provides `null`, so `next/head` falls
// back to mutating `document.head` directly via an effect.

import { createContext } from "react";

export const HeadManagerContext = createContext(null);
