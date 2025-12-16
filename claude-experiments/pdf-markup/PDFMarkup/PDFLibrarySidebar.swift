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
