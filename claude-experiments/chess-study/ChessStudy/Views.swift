import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel: StudyViewModel

    init(dataDir: String) {
        _viewModel = StateObject(wrappedValue: StudyViewModel(dataDir: dataDir))
    }

    var body: some View {
        NavigationStack {
            if viewModel.flashcards.isEmpty {
                CourseListView(viewModel: viewModel)
            } else {
                FlashcardView(viewModel: viewModel)
            }
        }
        .onAppear {
            viewModel.loadCourses()
        }
    }
}

struct CourseListView: View {
    @ObservedObject var viewModel: StudyViewModel

    var body: some View {
        List {
            if let error = viewModel.errorMessage {
                Text(error)
                    .foregroundColor(.red)
            }

            ForEach(viewModel.courses) { course in
                Button {
                    viewModel.selectCourse(course)
                } label: {
                    HStack {
                        Text(course.color == "W" ? "♔" : "♚")
                            .font(.title2)
                        VStack(alignment: .leading) {
                            Text(course.name)
                                .font(.headline)
                            Text(course.color == "W" ? "Playing as White" : "Playing as Black")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
            }
        }
        .navigationTitle("Chess Study")
        .overlay {
            if viewModel.isLoading {
                ProgressView()
            }
        }
    }
}

struct FlashcardView: View {
    @ObservedObject var viewModel: StudyViewModel
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Button("Back") {
                    viewModel.flashcards = []
                }
                Spacer()
                Text(viewModel.progress)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Button {
                    showSettings.toggle()
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
            .padding()

            if let card = viewModel.currentCard {
                Spacer()

                // Card content
                VStack(spacing: 20) {
                    // Context
                    Text(card.chapterName)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text(card.studyName)
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    // Line preview
                    Text(card.linePreview)
                        .font(.system(.title2, design: .monospaced))
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(12)

                    // Answer area
                    if viewModel.showAnswer {
                        let moveSquares = findMoveSquares(
                            beforeFen: card.positionFen,
                            afterFen: card.resultingFen
                        )

                        // Show the board with move highlighted
                        ChessBoardView(
                            fen: card.resultingFen,
                            highlightFrom: moveSquares.from,
                            highlightTo: moveSquares.to,
                            playerColor: viewModel.currentPlayerColor
                        )
                        .frame(maxWidth: 300, maxHeight: 300)
                        .padding(.vertical, 8)

                        Text(card.correctMove)
                            .font(.system(size: 36, weight: .bold, design: .monospaced))
                            .foregroundColor(.green)
                    } else {
                        Text("?")
                            .font(.system(size: 48, weight: .bold, design: .monospaced))
                            .foregroundColor(.secondary)
                            .frame(height: 300)
                    }
                }
                .padding()

                Spacer()

                // Buttons
                if viewModel.showAnswer {
                    HStack(spacing: 20) {
                        Button {
                            viewModel.markWrong()
                        } label: {
                            Label("Again", systemImage: "xmark.circle.fill")
                                .font(.title2)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red.opacity(0.2))
                                .foregroundColor(.red)
                                .cornerRadius(12)
                        }
                        .buttonStyle(.plain)

                        Button {
                            viewModel.markCorrect()
                        } label: {
                            Label("Got it", systemImage: "checkmark.circle.fill")
                                .font(.title2)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green.opacity(0.2))
                                .foregroundColor(.green)
                                .cornerRadius(12)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding()
                } else {
                    Button {
                        viewModel.reveal()
                    } label: {
                        Text("Show Answer")
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .buttonStyle(.plain)
                    .padding()
                }

                // Stats
                Text(viewModel.sessionStats)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom)

            } else {
                // Done!
                Spacer()
                VStack(spacing: 20) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.green)
                    Text("All Done!")
                        .font(.largeTitle)
                    Text(viewModel.sessionStats)
                        .foregroundColor(.secondary)
                    Button("Study Again") {
                        viewModel.restart()
                    }
                    .buttonStyle(.borderedProminent)
                }
                Spacer()
            }
        }
        #if os(iOS)
        .toolbar(.hidden, for: .navigationBar)
        #endif
        .sheet(isPresented: $showSettings) {
            SettingsSheet(viewModel: viewModel)
        }
    }
}

struct SettingsSheet: View {
    @ObservedObject var viewModel: StudyViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Move Depth") {
                    Stepper("Min: \(viewModel.minDepth)", value: $viewModel.minDepth, in: 1...viewModel.maxDepth)
                    Stepper("Max: \(viewModel.maxDepth)", value: $viewModel.maxDepth, in: viewModel.minDepth...30)
                }

                Section {
                    Button("Apply & Restart") {
                        viewModel.applyDepthFilter()
                        dismiss()
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
