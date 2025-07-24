import Cocoa
import WebKit

class ScoresPanelView: NSView {
    private let webView: WKWebView
    
    override init(frame frameRect: NSRect) {
        webView = WKWebView()
        
        super.init(frame: frameRect)
        
        setupView()
        setupWebView()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 0.9).cgColor
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        layer?.borderColor = NSColor(red: 0.3, green: 0.5, blue: 0.7, alpha: 1.0).cgColor
    }
    
    private func setupWebView() {
        webView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(webView)
        
        NSLayoutConstraint.activate([
            webView.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            webView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            webView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            webView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10)
        ])
        
        let htmlContent = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    background-color: transparent;
                    color: white;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                h2 {
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 20px;
                }
                ul {
                    list-style: none;
                    padding: 0;
                }
                li {
                    margin-bottom: 8px;
                    font-size: 14px;
                    font-weight: 500;
                }
                .status {
                    margin-top: 30px;
                    font-family: 'SF Mono', Monaco, monospace;
                    font-size: 11px;
                    color: #ccc;
                }
            </style>
        </head>
        <body>
            <h2>Scores</h2>
            <ul>
                <li>• Paul - 52444</li>
                <li>• Rob - 20345</li>
                <li>• Chris - 10345</li>
                <li>• John - 1395</li>
            </ul>
            <div class="status">
                GET :1 / took 5ms<br>
                GET :1 / took 56ms
            </div>
        </body>
        </html>
        """
        
        webView.loadHTMLString(htmlContent, baseURL: nil)
    }
}