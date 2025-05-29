//
//  ContentView.swift
//  ReadNever
//
//  Created by Jimmy Miller on 5/25/25.
//

import SwiftUI
import WebKit
import SwiftData
import UniformTypeIdentifiers

@Model
final class LinkItem {
    var id: UUID
    var urlString: String
    var title: String
    var createdAt: Date
    var category: String
    
    var url: URL? {
        URL(string: urlString)
    }
    
    init(url: URL, title: String, category: String = "Uncategorized") {
        self.id = UUID()
        self.urlString = url.absoluteString
        self.title = title
        self.createdAt = Date()
        self.category = category
    }
}

@Model
final class Category {
    var id: UUID
    var name: String
    var isExpanded: Bool
    
    init(name: String, isExpanded: Bool = true) {
        self.id = UUID()
        self.name = name
        self.isExpanded = isExpanded
    }
}

// WebViewController for stable WKWebView hosting
class WebViewController: NSViewController, WKNavigationDelegate {
    var webView: WKWebView!
    var url: URL? {
        didSet {
            loadURL()
        }
    }
    
    override func loadView() {
        let config = WKWebViewConfiguration()
        webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = self
        view = webView
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadURL()
    }
    
    private func loadURL() {
        guard let url = url, view != nil else { return }
        print("Loading URL: \(url.absoluteString)")
        let request = URLRequest(url: url)
        webView.load(request)
    }
    
    // MARK: - WKNavigationDelegate
    
    func webView(_ webView: WKWebView, didReceiveServerRedirectForProvisionalNavigation navigation: WKNavigation!) {
        print("Received server redirect: \(webView.url?.absoluteString ?? "unknown")")
    }
    
    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        print("Navigation response: \(navigationResponse.response.url?.absoluteString ?? "no URL")")
        
        // Handle HTTP redirects like 307
        if let httpResponse = navigationResponse.response as? HTTPURLResponse {
            print("HTTP status code: \(httpResponse.statusCode)")
            
            // Handle redirect status codes (301, 302, 303, 307, 308)
            if (300...399).contains(httpResponse.statusCode) {
                if let location = httpResponse.allHeaderFields["Location"] as? String,
                   let redirectURL = URL(string: location) {
                    print("Redirect location: \(location)")
                    
                    // Load the redirect URL manually
                    DispatchQueue.main.async {
                        webView.load(URLRequest(url: redirectURL))
                    }
                    decisionHandler(.cancel) // Cancel the current navigation
                    return
                }
            }
        }
        
        decisionHandler(.allow)
    }
    
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        // Log the navigation and always allow
        print("Navigation action: \(navigationAction.request.url?.absoluteString ?? "no URL")")
        decisionHandler(.allow)
    }
    
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        print("Navigation failed: \(error.localizedDescription)")
    }
    
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        print("Failed to load: \(error.localizedDescription)")
        
        // Try to handle common redirect and HTTP/HTTPS issues
        if let urlError = error as? URLError, let failingURL = urlError.failingURL {
            if failingURL.scheme == "http" {
                // Try to convert HTTP to HTTPS
                var components = URLComponents(url: failingURL, resolvingAgainstBaseURL: false)
                components?.scheme = "https"
                
                if let httpsURL = components?.url {
                    print("Retrying with HTTPS: \(httpsURL.absoluteString)")
                    webView.load(URLRequest(url: httpsURL))
                }
            }
        }
    }
}

// NSViewControllerRepresentable wrapper for WebViewController
struct WebView: NSViewControllerRepresentable {
    let url: URL
    
    func makeNSViewController(context: Context) -> WebViewController {
        let viewController = WebViewController()
        viewController.url = url
        return viewController
    }
    
    func updateNSViewController(_ nsViewController: WebViewController, context: Context) {
        if nsViewController.url != url {
            nsViewController.url = url
        }
    }
}

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Query private var links: [LinkItem]
    @Query private var categories: [Category]
    
    @State private var newLink: String = ""
    @State private var selectedLinkID: UUID? = nil
    @State private var isEditingCategory: Bool = false
    @State private var editingCategoryID: UUID? = nil
    @State private var newCategoryName: String = ""
    
    var body: some View {
        NavigationSplitView {
            VStack {
                // Link addition controls
                HStack {
                    TextField("Add link (https://...)", text: $newLink)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .onSubmit {
                            addLink()
                        }
                    Button("Add") {
                        addLink()
                    }
                }
                .padding()
                
                // Links list with collapsible sections
                List {
                    // Create a disclosure group for each category
                    ForEach(categories) { category in
                        DisclosureGroup(
                            isExpanded: Binding(
                                get: { category.isExpanded },
                                set: { category.isExpanded = $0 }
                            )
                        ) {
                            // Links in this category
                            ForEach(links.filter { $0.category == category.name }) { link in
                                Button(action: {
                                    selectedLinkID = link.id
                                }) {
                                    HStack {
                                        Text(link.title)
                                            .lineLimit(1)
                                        Spacer()
                                    }
                                    .padding(.vertical, 4)
                                }
                                .buttonStyle(PlainButtonStyle())
                                .background((selectedLinkID == link.id) ? Color.accentColor.opacity(0.2) : Color.clear)
                                .cornerRadius(4)
                                .contextMenu {
                                    // Move options
                                    ForEach(categories) { targetCategory in
                                        if targetCategory.name != category.name {
                                            Button("Move to \(targetCategory.name)") {
                                                moveLink(link, to: targetCategory.name)
                                            }
                                        }
                                    }
                                    Divider()
                                    Button("Delete", role: .destructive) {
                                        deleteLink(link)
                                    }
                                }
                                .draggable(link.id.uuidString)
                            }
                        } label: {
                            HStack {
                                if isEditingCategory && editingCategoryID == category.id {
                                    TextField("Category name", text: $newCategoryName)
                                        .onSubmit {
                                            if !newCategoryName.isEmpty {
                                                category.name = newCategoryName
                                                isEditingCategory = false
                                                editingCategoryID = nil
                                                newCategoryName = ""
                                            }
                                        }
                                } else {
                                    Text(category.name)
                                        .font(.headline)
                                }
                                Spacer()
                                Text("\(links.filter { $0.category == category.name }.count)")
                                    .foregroundColor(.secondary)
                                    .font(.caption)
                            }
                            .contextMenu {
                                if category.name != "Uncategorized" {
                                    Button("Rename") {
                                        newCategoryName = category.name
                                        isEditingCategory = true
                                        editingCategoryID = category.id
                                    }
                                    Button("Delete", role: .destructive) {
                                        deleteCategory(category)
                                    }
                                }
                            }
                        }
                        .dropDestination(for: String.self) { items, _ in
                            handleDrop(of: items, toCategory: category.name)
                        }
                    }
                }
                
                // Add category button at bottom of sidebar
                HStack {
                    Button(action: {
                        addCategory()
                    }) {
                        Label("Add Category", systemImage: "folder.badge.plus")
                    }
                    .buttonStyle(.borderless)
                    .padding(.horizontal)
                }
            }
            .frame(minWidth: 250)
            .onAppear {
                // Initialize with default categories if none exist
                if categories.isEmpty {
                    modelContext.insert(Category(name: "Reading"))
                    modelContext.insert(Category(name: "Tutorials"))
                    modelContext.insert(Category(name: "Uncategorized", isExpanded: false))
                }
            }
        } detail: {
            if let selectedID = selectedLinkID,
               let link = links.first(where: { $0.id == selectedID }),
               let url = link.url {
                WebView(url: url)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack {
                    Image(systemName: "link.circle")
                        .font(.system(size: 64))
                        .foregroundColor(.secondary)
                        .padding()
                    Text("Select a link to view")
                        .font(.title2)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
    
    private func addLink() {
        guard !newLink.isEmpty, let url = URL(string: newLink) else { return }
        
        // Extract domain name for title
        let title = url.host?.replacingOccurrences(of: "www.", with: "") ?? newLink
        
        // Default to first category, or "Uncategorized" if none exist
        let defaultCategory = categories.first?.name ?? "Uncategorized"
        
        let item = LinkItem(url: url, title: title, category: defaultCategory)
        modelContext.insert(item)
        newLink = ""
        
        // Create "Uncategorized" category if it doesn't exist yet
        if !categories.contains(where: { $0.name == "Uncategorized" }) {
            modelContext.insert(Category(name: "Uncategorized"))
        }
    }
    
    private func deleteLink(_ link: LinkItem) {
        modelContext.delete(link)
        if selectedLinkID == link.id {
            selectedLinkID = nil
        }
    }
    
    private func moveLink(_ link: LinkItem, to categoryName: String) {
        link.category = categoryName
    }
    
    private func handleDrop(of items: [String], toCategory: String) -> Bool {
        guard let item = items.first,
              let uuid = UUID(uuidString: item),
              let link = links.first(where: { $0.id == uuid }) else {
            return false
        }
        
        link.category = toCategory
        return true
    }
    
    private func addCategory() {
        let newName = "New Category"
        let uniqueName = makeUniqueNameForCategory(baseName: newName)
        modelContext.insert(Category(name: uniqueName))
    }
    
    private func deleteCategory(_ category: Category) {
        // Move all links in this category to "Uncategorized"
        for link in links.filter({ $0.category == category.name }) {
            link.category = "Uncategorized"
        }
        
        // Create "Uncategorized" if it doesn't exist
        if !categories.contains(where: { $0.name == "Uncategorized" }) {
            modelContext.insert(Category(name: "Uncategorized"))
        }
        
        // Delete the category
        modelContext.delete(category)
    }
    
    private func makeUniqueNameForCategory(baseName: String) -> String {
        var name = baseName
        var counter = 1
        
        while categories.contains(where: { $0.name == name }) {
            name = "\(baseName) \(counter)"
            counter += 1
        }
        
        return name
    }
}
