import SwiftUI

struct PDFLibrarySidebar: View {
    let library: PDFLibrary
    @Binding var selectedPDF: PDFMetadata?
    @State private var searchText = ""
    @State private var selectedFolder: String?
    @State private var expandedFolders: Set<String> = []

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
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search PDFs...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(.systemGray6))
            .cornerRadius(8)
            .padding()

            Divider()

            // Content
            if !searchText.isEmpty {
                // Search results
                searchResultsList
            } else {
                // Folder navigation
                folderList
            }
        }
        .frame(width: 300)
        .background(Color(.systemBackground))
    }

    var searchResultsList: some View {
        List(filteredPDFs, selection: $selectedPDF) { pdf in
            PDFListItem(pdf: pdf)
        }
        .listStyle(.sidebar)
    }

    var folderList: some View {
        List(selection: $selectedFolder) {
            ForEach(library.sortedFolders, id: \.self) { folder in
                Section {
                    if expandedFolders.contains(folder) {
                        ForEach(library.pdfs(in: folder)) { pdf in
                            PDFListItem(pdf: pdf)
                                .tag(pdf as PDFMetadata?)
                                .onTapGesture {
                                    selectedPDF = pdf
                                }
                        }
                    }
                } header: {
                    HStack {
                        Image(systemName: expandedFolders.contains(folder) ? "folder.fill" : "folder")
                        Text(folder)
                            .font(.headline)
                        Spacer()
                        Text("\(library.pdfs(in: folder).count)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        withAnimation {
                            if expandedFolders.contains(folder) {
                                expandedFolders.remove(folder)
                            } else {
                                expandedFolders.insert(folder)
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
    }
}

struct PDFListItem: View {
    let pdf: PDFMetadata

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(pdf.displayTitle)
                .font(.system(size: 13))
                .lineLimit(2)
            Text(pdf.displayAuthor)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            HStack {
                Image(systemName: "doc.text")
                    .font(.system(size: 9))
                Text("\(pdf.totalPages) pages")
                    .font(.system(size: 10))
            }
            .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}
