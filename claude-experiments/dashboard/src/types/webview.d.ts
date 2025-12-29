/**
 * TypeScript declarations for Electron webview tag
 */

declare namespace JSX {
  interface IntrinsicElements {
    webview: React.DetailedHTMLProps<
      React.HTMLAttributes<HTMLWebViewElement> & {
        src?: string;
        preload?: string;
        partition?: string;
        allowpopups?: boolean;
        webpreferences?: string;
        httpreferrer?: string;
        useragent?: string;
        disablewebsecurity?: boolean;
        nodeintegration?: boolean;
        nodeintegrationinsubframes?: boolean;
        plugins?: boolean;
        enableremotemodule?: boolean;
      },
      HTMLWebViewElement
    >;
  }
}

interface HTMLWebViewElement extends HTMLElement {
  src: string;
  loadURL(url: string): void;
  getURL(): string;
  getTitle(): string;
  isLoading(): boolean;
  isLoadingMainFrame(): boolean;
  isWaitingForResponse(): boolean;
  stop(): void;
  reload(): void;
  reloadIgnoringCache(): void;
  canGoBack(): boolean;
  canGoForward(): boolean;
  canGoToOffset(offset: number): boolean;
  goBack(): void;
  goForward(): void;
  goToIndex(index: number): void;
  goToOffset(offset: number): void;
  isCrashed(): boolean;
  setUserAgent(userAgent: string): void;
  getUserAgent(): string;
  insertCSS(css: string): Promise<string>;
  executeJavaScript(code: string): Promise<unknown>;
  openDevTools(): void;
  closeDevTools(): void;
  isDevToolsOpened(): boolean;
  isDevToolsFocused(): boolean;
  inspectElement(x: number, y: number): void;
}
