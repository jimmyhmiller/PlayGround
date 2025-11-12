import SwiftUI

struct CategorySidebar: View {
    @ObservedObject var categoryManager: CategoryManager
    let onCategorySelected: (String) -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Header
            Text("Categories")
                .font(.headline)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Category List
            ScrollView {
                VStack(spacing: 12) {
                    ForEach(categoryManager.categories, id: \.self) { category in
                        CategoryButton(
                            title: category,
                            action: {
                                onCategorySelected(category)
                            }
                        )
                    }

                    // Add Category Button
                    Button(action: {
                        categoryManager.showingAddCategory = true
                    }) {
                        HStack {
                            Image(systemName: "plus.circle.fill")
                            Text("Add Category")
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 60)
                        .background(Color.accentColor.opacity(0.1))
                        .foregroundColor(.accentColor)
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
                .padding()
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
        .sheet(isPresented: $categoryManager.showingAddCategory) {
            AddCategorySheet(categoryManager: categoryManager)
        }
    }
}

struct CategoryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .medium))
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
                .frame(minHeight: 60)
                .padding(.horizontal, 8)
                .background(Color.accentColor)
                .foregroundColor(.white)
                .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
}

struct AddCategorySheet: View {
    @ObservedObject var categoryManager: CategoryManager
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 20) {
            Text("Add New Category")
                .font(.title2)
                .bold()

            TextField("Category Name", text: $categoryManager.newCategoryName)
                .textFieldStyle(.roundedBorder)
                .focused($isFocused)
                .onSubmit {
                    addCategory()
                }

            HStack {
                Button("Cancel") {
                    categoryManager.showingAddCategory = false
                    categoryManager.newCategoryName = ""
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Add") {
                    addCategory()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(categoryManager.newCategoryName.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(30)
        .frame(width: 400)
        .onAppear {
            isFocused = true
        }
    }

    private func addCategory() {
        categoryManager.addCategory(categoryManager.newCategoryName)
        categoryManager.showingAddCategory = false
        categoryManager.newCategoryName = ""
    }
}
