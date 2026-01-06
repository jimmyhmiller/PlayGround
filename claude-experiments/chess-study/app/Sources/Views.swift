import SwiftUI

// MARK: - Platform Colors

#if os(iOS)
let backgroundColor = Color(.systemGroupedBackground)
let cardBackgroundColor = Color(.systemBackground)
#else
let backgroundColor = Color(nsColor: .windowBackgroundColor)
let cardBackgroundColor = Color(nsColor: .controlBackgroundColor)
#endif

struct ContentView: View {
    @StateObject private var viewModel: StudyViewModel

    init(dataDir: String) {
        _viewModel = StateObject(wrappedValue: StudyViewModel(dataDir: dataDir))
    }

    var body: some View {
        NavigationStack {
            switch viewModel.studyState {
            case .courseList:
                CourseListView(viewModel: viewModel)
            case .chapterSelection:
                ChapterSelectionView(viewModel: viewModel)
            case .studying:
                FlashcardView(viewModel: viewModel)
            }
        }
        .onAppear {
            viewModel.loadCourses()
        }
    }
}

// MARK: - Course List

struct CourseListView: View {
    @ObservedObject var viewModel: StudyViewModel

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if let error = viewModel.errorMessage {
                    ErrorBanner(message: error)
                }

                ForEach(viewModel.courses) { course in
                    CourseCard(course: course) {
                        withAnimation(.spring(response: 0.3)) {
                            viewModel.selectCourse(course)
                        }
                    }
                }
            }
            .padding()
        }
        .background(backgroundColor)
        .navigationTitle("Chess Study")
        .overlay {
            if viewModel.isLoading {
                ProgressView()
                    .scaleEffect(1.2)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.ultraThinMaterial)
            }
        }
    }
}

struct CourseCard: View {
    let course: Course
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(course.color == "W" ? Color.white : Color.black)
                        .frame(width: 50, height: 50)
                        .shadow(color: .black.opacity(0.1), radius: 2, y: 1)

                    Text(course.color == "W" ? "♔" : "♚")
                        .font(.system(size: 28))
                        .foregroundColor(course.color == "W" ? .black : .white)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(course.name)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text(course.color == "W" ? "Playing as White" : "Playing as Black")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.secondary.opacity(0.5))
            }
            .padding()
            .background(cardBackgroundColor)
            .cornerRadius(16)
            .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
        }
        .buttonStyle(.plain)
    }
}

struct ErrorBanner: View {
    let message: String

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
            Text(message)
                .font(.subheadline)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Chapter Selection

struct ChapterSelectionView: View {
    @ObservedObject var viewModel: StudyViewModel

    var body: some View {
        ZStack {
            backgroundColor.ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                HStack {
                    Button {
                        withAnimation(.spring(response: 0.3)) {
                            viewModel.backToCourses()
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 16, weight: .semibold))
                            Text("Courses")
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.accentColor)
                    }

                    Spacer()

                    HStack(spacing: 16) {
                        Button("None") {
                            viewModel.deselectAllChapters()
                        }
                        .font(.subheadline)
                        .foregroundColor(.accentColor)

                        Button("All") {
                            viewModel.selectAllChapters()
                        }
                        .font(.subheadline)
                        .foregroundColor(.accentColor)
                    }
                }
                .padding()

                // Chapter list
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(viewModel.chapters) { chapter in
                            ChapterRow(
                                chapter: chapter,
                                isSelected: viewModel.selectedChapterIds.contains(chapter.id)
                            ) {
                                viewModel.toggleChapter(chapter)
                            }
                        }
                    }
                    .padding(.horizontal)
                }

                // Bottom action
                VStack(spacing: 12) {
                    let selectedCount = viewModel.selectedChapterIds.count
                    Text("\(selectedCount) chapter\(selectedCount == 1 ? "" : "s") selected")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Button {
                        withAnimation(.spring(response: 0.3)) {
                            viewModel.startStudying()
                        }
                    } label: {
                        Text("Start Studying")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .foregroundColor(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 14)
                                    .fill(viewModel.selectedChapterIds.isEmpty ? Color.gray : Color.accentColor)
                            )
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.selectedChapterIds.isEmpty)
                }
                .padding()
                .background(backgroundColor)
            }
        }
        #if os(iOS)
        .toolbar(.hidden, for: .navigationBar)
        #endif
        .overlay {
            if viewModel.isLoading {
                ProgressView()
                    .scaleEffect(1.2)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.ultraThinMaterial)
            }
        }
    }
}

struct ChapterRow: View {
    let chapter: Chapter
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 22))
                    .foregroundColor(isSelected ? .accentColor : .secondary.opacity(0.4))

                Text(chapter.name)
                    .font(.body)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.leading)

                Spacer()
            }
            .padding()
            .background(cardBackgroundColor)
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Flashcard View

struct FlashcardView: View {
    @ObservedObject var viewModel: StudyViewModel
    @State private var showSettings = false

    var body: some View {
        ZStack {
            backgroundColor
                .ignoresSafeArea()

            VStack(spacing: 0) {
                FlashcardHeader(
                    progress: viewModel.progress,
                    onBack: {
                        withAnimation(.spring(response: 0.3)) {
                            viewModel.backToChapters()
                        }
                    },
                    onSettings: { showSettings.toggle() }
                )

                if let card = viewModel.currentCard {
                    ScrollView {
                        VStack(spacing: 12) {
                            VStack(spacing: 4) {
                                Text(card.chapterName)
                                    .font(.caption2)
                                    .fontWeight(.medium)
                                    .foregroundColor(.secondary)
                                    .textCase(.uppercase)
                                    .tracking(0.5)

                                Text(card.studyName)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Text(card.linePreview)
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(.primary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 10)
                                .frame(maxWidth: .infinity)
                                .background(cardBackgroundColor)
                                .cornerRadius(10)
                                .shadow(color: .black.opacity(0.04), radius: 4, y: 2)

                            if viewModel.showAnswer {
                                AnswerRevealView(card: card, playerColor: viewModel.currentPlayerColor)
                                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
                            } else {
                                QuestionPlaceholder()
                                    .transition(.opacity)
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }

                    VStack(spacing: 8) {
                        if viewModel.showAnswer {
                            AnswerButtons(
                                onAgain: {
                                    withAnimation(.spring(response: 0.35)) {
                                        viewModel.markWrong()
                                    }
                                },
                                onGotIt: {
                                    withAnimation(.spring(response: 0.35)) {
                                        viewModel.markCorrect()
                                    }
                                }
                            )
                        } else {
                            ShowAnswerButton {
                                withAnimation(.spring(response: 0.4)) {
                                    viewModel.reveal()
                                }
                            }
                        }

                        Text(viewModel.sessionStats)
                            .font(.caption2)
                            .foregroundColor(.secondary.opacity(0.7))
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                    .background(
                        backgroundColor
                            .shadow(color: .black.opacity(0.05), radius: 6, y: -3)
                            .ignoresSafeArea(edges: .bottom)
                    )

                } else {
                    CompletionView(
                        stats: viewModel.sessionStats,
                        onRestart: { viewModel.restart() },
                        onBackToChapters: {
                            withAnimation(.spring(response: 0.3)) {
                                viewModel.backToChapters()
                            }
                        }
                    )
                }
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

struct FlashcardHeader: View {
    let progress: String
    let onBack: () -> Void
    let onSettings: () -> Void

    var body: some View {
        HStack {
            Button(action: onBack) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 16, weight: .semibold))
                    Text("Chapters")
                        .fontWeight(.medium)
                }
                .foregroundColor(.accentColor)
            }

            Spacer()

            Text(progress)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(cardBackgroundColor)
                .cornerRadius(20)
                .shadow(color: .black.opacity(0.04), radius: 2, y: 1)

            Spacer()

            Button(action: onSettings) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundColor(.accentColor)
                    .frame(width: 44, height: 44)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

struct QuestionPlaceholder: View {
    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(cardBackgroundColor)
                    .frame(width: 80, height: 80)
                    .shadow(color: .black.opacity(0.06), radius: 6, y: 3)

                Text("?")
                    .font(.system(size: 40, weight: .bold, design: .rounded))
                    .foregroundColor(.secondary.opacity(0.5))
            }

            Text("What's the best move?")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(height: 120)
    }
}

struct AnswerRevealView: View {
    let card: Flashcard
    let playerColor: String
    @State private var currentMoveIndex: Int

    init(card: Flashcard, playerColor: String) {
        self.card = card
        self.playerColor = playerColor
        _currentMoveIndex = State(initialValue: card.moveSequence.count - 1)
    }

    private var startingFen: String {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }

    private var currentFen: String {
        if currentMoveIndex < 0 {
            return startingFen
        }
        return card.moveSequence[currentMoveIndex].resultingFen
    }

    private var highlightSquares: (from: String?, to: String?) {
        if currentMoveIndex < 0 {
            return (nil, nil)
        }
        let step = card.moveSequence[currentMoveIndex]
        return findMoveSquares(beforeFen: step.fen, afterFen: step.resultingFen)
    }

    private var isAtStart: Bool { currentMoveIndex < 0 }
    private var isAtEnd: Bool { currentMoveIndex >= card.moveSequence.count - 1 }

    var body: some View {
        VStack(spacing: 8) {
            // Board - smaller and responsive
            GeometryReader { geo in
                let size = min(geo.size.width, 280)
                ChessBoardView(
                    fen: currentFen,
                    highlightFrom: highlightSquares.from,
                    highlightTo: highlightSquares.to,
                    playerColor: playerColor
                )
                .frame(width: size, height: size)
                .background(cardBackgroundColor)
                .cornerRadius(12)
                .shadow(color: .black.opacity(0.08), radius: 8, y: 3)
                .frame(maxWidth: .infinity)
            }
            .frame(height: 280)

            // Navigation controls
            HStack(spacing: 8) {
                Button { currentMoveIndex = -1 } label: {
                    Image(systemName: "backward.end.fill")
                        .font(.system(size: 14))
                        .foregroundColor(isAtStart ? .secondary.opacity(0.3) : .accentColor)
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isAtStart)

                Button { if currentMoveIndex > -1 { currentMoveIndex -= 1 } } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(isAtStart ? .secondary.opacity(0.3) : .accentColor)
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isAtStart)

                Text(currentMoveIndex < 0 ? "Start" : card.moveSequence[currentMoveIndex].notation)
                    .font(.system(size: 12, design: .monospaced))
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                    .frame(minWidth: 50)

                Button { if currentMoveIndex < card.moveSequence.count - 1 { currentMoveIndex += 1 } } label: {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(isAtEnd ? .secondary.opacity(0.3) : .accentColor)
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isAtEnd)

                Button { currentMoveIndex = card.moveSequence.count - 1 } label: {
                    Image(systemName: "forward.end.fill")
                        .font(.system(size: 14))
                        .foregroundColor(isAtEnd ? .secondary.opacity(0.3) : .accentColor)
                        .frame(width: 32, height: 32)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isAtEnd)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(cardBackgroundColor)
            .cornerRadius(8)

            // Answer
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.system(size: 18))

                Text(card.correctMove)
                    .font(.system(size: 22, weight: .bold, design: .monospaced))
                    .foregroundColor(.primary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
        }
    }
}

struct AnswerButtons: View {
    let onAgain: () -> Void
    let onGotIt: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Button(action: onAgain) {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Again")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .foregroundColor(.red)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.red.opacity(0.4), lineWidth: 1.5)
                )
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button(action: onGotIt) {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark")
                        .font(.system(size: 15, weight: .semibold))
                    Text("Got it")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .foregroundColor(.white)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.green)
                )
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}

struct ShowAnswerButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text("Show Answer")
                .font(.subheadline)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .foregroundColor(.white)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.accentColor)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct CompletionView: View {
    let stats: String
    let onRestart: () -> Void
    let onBackToChapters: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.1))
                    .frame(width: 120, height: 120)

                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(.green)
            }

            VStack(spacing: 8) {
                Text("All Done!")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text(stats)
                    .font(.title3)
                    .foregroundColor(.secondary)
            }

            VStack(spacing: 12) {
                Button(action: onRestart) {
                    HStack(spacing: 8) {
                        Image(systemName: "arrow.counterclockwise")
                        Text("Study Again")
                    }
                    .font(.headline)
                    .frame(maxWidth: 200)
                    .padding(.vertical, 14)
                    .foregroundColor(.white)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color.accentColor)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Button(action: onBackToChapters) {
                    Text("Change Chapters")
                        .font(.subheadline)
                        .foregroundColor(.accentColor)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 16)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .padding(.top, 8)

            Spacer()
        }
    }
}

// MARK: - Settings Sheet

struct SettingsSheet: View {
    @ObservedObject var viewModel: StudyViewModel
    @Environment(\.dismiss) var dismiss
    @State private var showResetConfirm = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Move Depth Range")
                            .font(.headline)

                        VStack(spacing: 20) {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Minimum: Move \(viewModel.minDepth)")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Stepper("", value: $viewModel.minDepth, in: 1...viewModel.maxDepth)
                                    .labelsHidden()
                            }

                            VStack(alignment: .leading, spacing: 8) {
                                Text("Maximum: Move \(viewModel.maxDepth)")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Stepper("", value: $viewModel.maxDepth, in: viewModel.minDepth...30)
                                    .labelsHidden()
                            }
                        }
                    }
                    .padding(.vertical, 8)
                } footer: {
                    Text("Only practice moves within this range of the opening.")
                }

                Section {
                    Toggle("Include Mastered Cards", isOn: $viewModel.includeMastered)
                } footer: {
                    Text("Show cards you've already mastered in the review session.")
                }

                Section {
                    Button {
                        viewModel.applyFilters()
                        dismiss()
                    } label: {
                        HStack {
                            Spacer()
                            Text("Apply & Restart")
                                .fontWeight(.semibold)
                            Spacer()
                        }
                    }
                }

                Section {
                    Button(role: .destructive) {
                        showResetConfirm = true
                    } label: {
                        HStack {
                            Spacer()
                            Text("Reset All Progress")
                            Spacer()
                        }
                    }
                } footer: {
                    Text("This will mark all cards as unlearned for this course.")
                }
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .confirmationDialog("Reset Progress?", isPresented: $showResetConfirm, titleVisibility: .visible) {
                Button("Reset All Progress", role: .destructive) {
                    viewModel.resetProgress()
                    dismiss()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will mark all cards as unlearned. You'll need to review them again.")
            }
        }
        .presentationDetents([.medium, .large])
    }
}
