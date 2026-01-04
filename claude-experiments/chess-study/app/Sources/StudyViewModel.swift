import SwiftUI

@MainActor
class StudyViewModel: ObservableObject {
    @Published var courses: [Course] = []
    @Published var flashcards: [Flashcard] = []
    @Published var currentIndex: Int = 0
    @Published var showAnswer: Bool = false
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    @Published var minDepth: Int = 1
    @Published var maxDepth: Int = 20
    @Published var currentPlayerColor: String = "W"

    private var remainingCards: [Flashcard] = []
    private var correctCount: Int = 0
    private var totalAttempts: Int = 0
    private var currentCourseId: String?

    let dataLoader: DataLoader

    private let depthSettingsKey = "courseDepthSettings"

    var currentCard: Flashcard? {
        guard currentIndex < remainingCards.count else { return nil }
        return remainingCards[currentIndex]
    }

    var progress: String {
        let remaining = remainingCards.count
        let total = flashcards.count
        return "\(total - remaining)/\(total) mastered"
    }

    var sessionStats: String {
        return "\(correctCount)/\(totalAttempts) this session"
    }

    var depthRange: ClosedRange<Int> {
        let min = flashcards.map { $0.moveNumber }.min() ?? 1
        let max = flashcards.map { $0.moveNumber }.max() ?? 20
        return min...max
    }

    init(dataDir: String) {
        self.dataLoader = DataLoader(dataDir: dataDir)
    }

    func loadCourses() {
        isLoading = true
        errorMessage = nil

        do {
            courses = try dataLoader.loadCourses()
        } catch {
            errorMessage = "Failed to load courses: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func selectCourse(_ course: Course) {
        isLoading = true
        errorMessage = nil
        currentPlayerColor = course.color
        currentCourseId = course.id

        // Load persisted depth settings for this course
        let (savedMin, savedMax) = loadDepthSettings(for: course.id)
        minDepth = savedMin
        maxDepth = savedMax

        do {
            flashcards = try dataLoader.loadAllFlashcards(for: course)
            applyDepthFilter()
        } catch {
            errorMessage = "Failed to load flashcards: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func applyDepthFilter() {
        let filtered = flashcards.filter { $0.moveNumber >= minDepth && $0.moveNumber <= maxDepth }
        remainingCards = filtered.shuffled()
        currentIndex = 0
        showAnswer = false
        correctCount = 0
        totalAttempts = 0

        // Save depth settings for this course
        if let courseId = currentCourseId {
            saveDepthSettings(for: courseId, min: minDepth, max: maxDepth)
        }
    }

    // MARK: - Persistence

    private func loadDepthSettings(for courseId: String) -> (min: Int, max: Int) {
        guard let allSettings = UserDefaults.standard.dictionary(forKey: depthSettingsKey),
              let courseSettings = allSettings[courseId] as? [String: Int],
              let min = courseSettings["min"],
              let max = courseSettings["max"] else {
            return (1, 20)  // Default values
        }
        return (min, max)
    }

    private func saveDepthSettings(for courseId: String, min: Int, max: Int) {
        var allSettings = UserDefaults.standard.dictionary(forKey: depthSettingsKey) ?? [:]
        allSettings[courseId] = ["min": min, "max": max]
        UserDefaults.standard.set(allSettings, forKey: depthSettingsKey)
    }

    func reveal() {
        showAnswer = true
    }

    func markCorrect() {
        totalAttempts += 1
        correctCount += 1
        // Remove the card (mastered)
        if currentIndex < remainingCards.count {
            remainingCards.remove(at: currentIndex)
        }
        // Keep same index (next card slides into place)
        if currentIndex >= remainingCards.count {
            currentIndex = 0
        }
        showAnswer = false
    }

    func markWrong() {
        totalAttempts += 1
        // Move card to back of deck
        if currentIndex < remainingCards.count {
            let card = remainingCards.remove(at: currentIndex)
            remainingCards.append(card)
        }
        if currentIndex >= remainingCards.count {
            currentIndex = 0
        }
        showAnswer = false
    }

    func restart() {
        applyDepthFilter()
    }
}
