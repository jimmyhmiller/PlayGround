import SwiftUI
import PDFKit

struct PDFLibrarySidebar: View {
    let library: PDFLibrary
    @Binding var selectedPDF: PDFMetadata?
    var onSelectSharedPDF: ((String, PDFDocument) -> Void)?
    @State private var searchText = ""
    @State private var selectedFolder: String?
    @State private var expandedFolders: Set<String> = []
    @State private var recentlyHighlightedExpanded = true
    @State private var recentlyHighlighted: [(hash: String, lastModified: Date)] = []

    var filteredPDFs: [PDFMetadata] {
        if !searchText.isEmpty {
            return library.search(query: searchText)
        } else if let folder = selectedFolder {
            return library.pdfs(in: folder)
        } else {
            return []
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(Color.secondary.opacity(0.6))
                    .font(.system(size: 14))
                TextField("Search", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(Color.secondary.opacity(0.6))
                            .font(.system(size: 14))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color(.systemGray6).opacity(0.8))
            .cornerRadius(10)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)

            // Content
            if !searchText.isEmpty {
                searchResultsList
            } else {
                folderList
            }
        }
        .frame(minWidth: 260, maxWidth: 320)
        .onAppear {
            loadRecentlyHighlighted()
        }
    }

    private func loadRecentlyHighlighted() {
        recentlyHighlighted = DrawingManager.shared.getRecentlyHighlightedHashes(limit: 10)
    }

    var searchResultsList: some View {
        List(filteredPDFs, selection: $selectedPDF) { pdf in
            PDFListItem(pdf: pdf)
        }
        .listStyle(.sidebar)
    }

    var folderList: some View {
        List(selection: $selectedFolder) {
            // Recently Highlighted section
            if !recentlyHighlighted.isEmpty {
                DisclosureGroup(isExpanded: $recentlyHighlightedExpanded) {
                    ForEach(recentlyHighlighted, id: \.hash) { item in
                        RecentlyHighlightedItem(
                            hash: item.hash,
                            lastModified: item.lastModified,
                            library: library,
                            isSelected: selectedPDF?.hash == item.hash,
                            onSelect: { pdf in
                                selectedPDF = pdf
                            },
                            onSelectShared: onSelectSharedPDF
                        )
                    }
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "highlighter")
                            .foregroundColor(.orange)
                            .font(.system(size: 14))
                        Text("Recently Highlighted")
                            .font(.system(size: 13, weight: .semibold))
                        Spacer()
                        Text("\(recentlyHighlighted.count)")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color(.systemGray5))
                            .cornerRadius(8)
                    }
                }
            }

            // Folder sections
            ForEach(library.sortedFolders, id: \.self) { folder in
                DisclosureGroup(
                    isExpanded: folderBinding(for: folder)
                ) {
                    ForEach(library.pdfs(in: folder)) { pdf in
                        PDFListItem(pdf: pdf, isSelected: selectedPDF?.id == pdf.id)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                selectedPDF = pdf
                            }
                    }
                } label: {
                    FolderLabel(name: folder, count: library.pdfs(in: folder).count)
                }
            }
        }
        .listStyle(.sidebar)
        .refreshable {
            loadRecentlyHighlighted()
        }
    }

    private func folderBinding(for folder: String) -> Binding<Bool> {
        Binding(
            get: { expandedFolders.contains(folder) },
            set: { isExpanded in
                if isExpanded {
                    expandedFolders.insert(folder)
                } else {
                    expandedFolders.remove(folder)
                }
            }
        )
    }
}

struct FolderLabel: View {
    let name: String
    let count: Int

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "folder.fill")
                .foregroundColor(.accentColor)
                .font(.system(size: 14))
            Text(name)
                .font(.system(size: 13, weight: .semibold))
            Spacer()
            Text("\(count)")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(.systemGray5))
                .cornerRadius(8)
        }
    }
}

struct PDFListItem: View {
    let pdf: PDFMetadata
    var isSelected: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(pdf.displayTitle)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(2)
                .foregroundColor(.primary)
            if !pdf.displayAuthor.isEmpty && pdf.displayAuthor != "Unknown" {
                Text(pdf.displayAuthor)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            Text("\(pdf.totalPages) pages")
                .font(.system(size: 10))
                .foregroundColor(Color.secondary.opacity(0.7))
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
        )
    }
}

struct RecentlyHighlightedItem: View {
    let hash: String
    let lastModified: Date
    let library: PDFLibrary
    var isSelected: Bool = false
    var onSelect: ((PDFMetadata) -> Void)?
    var onSelectShared: ((String, PDFDocument) -> Void)?

    private var pdf: PDFMetadata? {
        library.pdfs.first { $0.hash == hash }
    }

    private var timeAgo: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: lastModified, relativeTo: Date())
    }

    var body: some View {
        if let pdf = pdf {
            // PDF is in the library
            VStack(alignment: .leading, spacing: 3) {
                Text(pdf.displayTitle)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(2)
                    .foregroundColor(.primary)
                HStack {
                    if !pdf.displayAuthor.isEmpty && pdf.displayAuthor != "Unknown" {
                        Text(pdf.displayAuthor)
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    Text(timeAgo)
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                }
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
            )
            .contentShape(Rectangle())
            .onTapGesture {
                onSelect?(pdf)
            }
        } else {
            // PDF might be a shared one not in the library
            SharedPDFItem(hash: hash, timeAgo: timeAgo, isSelected: isSelected, onSelectShared: onSelectShared)
        }
    }
}

struct SharedPDFItem: View {
    let hash: String
    let timeAgo: String
    var isSelected: Bool = false
    var onSelectShared: ((String, PDFDocument) -> Void)?

    private var sharedPDFInfo: (url: URL, fileName: String)? {
        let fileManager = FileManager.default
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let sharedPDFsDir = documentsDir.appendingPathComponent("SharedPDFs")

        guard let files = try? fileManager.contentsOfDirectory(atPath: sharedPDFsDir.path) else {
            return nil
        }

        // Find file that starts with our hash
        if let file = files.first(where: { $0.hasPrefix(hash) }) {
            let url = sharedPDFsDir.appendingPathComponent(file)
            // Extract original filename (after the hash_)
            let parts = file.components(separatedBy: "_")
            let fileName = parts.count > 1 ? parts.dropFirst().joined(separator: "_") : file
            return (url, fileName)
        }
        return nil
    }

    var body: some View {
        if let info = sharedPDFInfo {
            VStack(alignment: .leading, spacing: 3) {
                Text(info.fileName.replacingOccurrences(of: ".pdf", with: ""))
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(2)
                    .foregroundColor(.primary)
                HStack {
                    Text("Shared PDF")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(timeAgo)
                        .font(.system(size: 10))
                        .foregroundColor(.orange)
                }
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
            )
            .contentShape(Rectangle())
            .onTapGesture {
                if let document = PDFDocument(url: info.url) {
                    onSelectShared?(hash, document)
                }
            }
        } else {
            // PDF no longer available
            VStack(alignment: .leading, spacing: 3) {
                Text("PDF not found")
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(2)
                    .foregroundColor(.secondary)
                Text(timeAgo)
                    .font(.system(size: 10))
                    .foregroundColor(.orange)
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 4)
            .opacity(0.5)
        }
    }
}
