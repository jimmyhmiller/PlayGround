import SwiftUI

enum StudyState {
    case courseList
    case chapterSelection
    case studying
}

@MainActor
class StudyViewModel: ObservableObject {
    @Published var courses: [Course] = []
    @Published var chapters: [Chapter] = []
    @Published var selectedChapterIds: Set<String> = []
    @Published var flashcards: [Flashcard] = []
    @Published var currentIndex: Int = 0
    @Published var showAnswer: Bool = false
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var studyState: StudyState = .courseList

    @Published var minDepth: Int = 1
    @Published var maxDepth: Int = 20
    @Published var currentPlayerColor: String = "W"
    @Published var includeMastered: Bool = false

    private var remainingCards: [Flashcard] = []
    private var sessionCards: [Flashcard] = []  // Cards for this session (after depth filter)
    private var correctCount: Int = 0
    private var totalAttempts: Int = 0
    private var currentCourse: Course?
    private var masteredCardIds: Set<String> = []

    let dataLoader: DataLoader

    private let depthSettingsKey = "courseDepthSettings"
    private let chapterSettingsKey = "courseChapterSettings"
    private let masteredCardsKey = "masteredCards"

    var currentCard: Flashcard? {
        guard currentIndex < remainingCards.count else { return nil }
        return remainingCards[currentIndex]
    }

    var progress: String {
        let remaining = remainingCards.count
        let total = sessionCards.count
        return "\(remaining) left"
    }

    var sessionStats: String {
        return "\(correctCount)/\(totalAttempts) this session"
    }

    var masteredCount: Int {
        guard let course = currentCourse else { return 0 }
        return loadMasteredCards(for: course.id).count
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
        currentCourse = course
        currentPlayerColor = course.color

        // Load persisted settings for this course
        let (savedMin, savedMax) = loadDepthSettings(for: course.id)
        minDepth = savedMin
        maxDepth = savedMax
        selectedChapterIds = loadSelectedChapters(for: course.id)
        masteredCardIds = loadMasteredCards(for: course.id)

        do {
            chapters = try dataLoader.loadChapters(courseId: course.id)
            // If no chapters selected yet, select all
            if selectedChapterIds.isEmpty {
                selectedChapterIds = Set(chapters.map { $0.id })
            }
            studyState = .chapterSelection
        } catch {
            errorMessage = "Failed to load chapters: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func toggleChapter(_ chapter: Chapter) {
        if selectedChapterIds.contains(chapter.id) {
            selectedChapterIds.remove(chapter.id)
        } else {
            selectedChapterIds.insert(chapter.id)
        }
    }

    func selectAllChapters() {
        selectedChapterIds = Set(chapters.map { $0.id })
    }

    func deselectAllChapters() {
        selectedChapterIds.removeAll()
    }

    func startStudying() {
        guard let course = currentCourse else { return }

        isLoading = true
        errorMessage = nil

        // Save selected chapters
        saveSelectedChapters(for: course.id, chapterIds: selectedChapterIds)

        do {
            flashcards = try dataLoader.loadAllFlashcards(for: course, chapterIds: selectedChapterIds)
            applyFilters()
            studyState = .studying
        } catch {
            errorMessage = "Failed to load flashcards: \(error.localizedDescription)"
        }

        isLoading = false
    }

    func applyFilters() {
        // First filter by depth
        let depthFiltered = flashcards.filter { $0.moveNumber >= minDepth && $0.moveNumber <= maxDepth }
        sessionCards = depthFiltered

        // Then filter out mastered cards unless includeMastered is true
        var filtered = depthFiltered
        if !includeMastered {
            filtered = filtered.filter { !masteredCardIds.contains($0.id) }
        }

        remainingCards = filtered.shuffled()
        currentIndex = 0
        showAnswer = false
        correctCount = 0
        totalAttempts = 0

        // Save depth settings for this course
        if let course = currentCourse {
            saveDepthSettings(for: course.id, min: minDepth, max: maxDepth)
        }
    }

    // Restart session including cards mastered this session
    func restartSession() {
        remainingCards = sessionCards.shuffled()
        currentIndex = 0
        showAnswer = false
        correctCount = 0
        totalAttempts = 0
    }

    func backToChapters() {
        studyState = .chapterSelection
        flashcards = []
        remainingCards = []
        sessionCards = []
    }

    func backToCourses() {
        studyState = .courseList
        chapters = []
        flashcards = []
        remainingCards = []
        sessionCards = []
        currentCourse = nil
    }

    // MARK: - Persistence

    private func loadDepthSettings(for courseId: String) -> (min: Int, max: Int) {
        guard let allSettings = UserDefaults.standard.dictionary(forKey: depthSettingsKey),
              let courseSettings = allSettings[courseId] as? [String: Int],
              let min = courseSettings["min"],
              let max = courseSettings["max"] else {
            return (1, 20)
        }
        return (min, max)
    }

    private func saveDepthSettings(for courseId: String, min: Int, max: Int) {
        var allSettings = UserDefaults.standard.dictionary(forKey: depthSettingsKey) ?? [:]
        allSettings[courseId] = ["min": min, "max": max]
        UserDefaults.standard.set(allSettings, forKey: depthSettingsKey)
    }

    private func loadSelectedChapters(for courseId: String) -> Set<String> {
        guard let allSettings = UserDefaults.standard.dictionary(forKey: chapterSettingsKey),
              let chapterIds = allSettings[courseId] as? [String] else {
            return []
        }
        return Set(chapterIds)
    }

    private func saveSelectedChapters(for courseId: String, chapterIds: Set<String>) {
        var allSettings = UserDefaults.standard.dictionary(forKey: chapterSettingsKey) ?? [:]
        allSettings[courseId] = Array(chapterIds)
        UserDefaults.standard.set(allSettings, forKey: chapterSettingsKey)
    }

    private func loadMasteredCards(for courseId: String) -> Set<String> {
        guard let allSettings = UserDefaults.standard.dictionary(forKey: masteredCardsKey),
              let cardIds = allSettings[courseId] as? [String] else {
            return []
        }
        return Set(cardIds)
    }

    private func saveMasteredCards(for courseId: String, cardIds: Set<String>) {
        var allSettings = UserDefaults.standard.dictionary(forKey: masteredCardsKey) ?? [:]
        allSettings[courseId] = Array(cardIds)
        UserDefaults.standard.set(allSettings, forKey: masteredCardsKey)
    }

    func reveal() {
        showAnswer = true
    }

    func markCorrect() {
        totalAttempts += 1
        correctCount += 1

        // Persist as mastered
        if let card = currentCard, let course = currentCourse {
            masteredCardIds.insert(card.id)
            saveMasteredCards(for: course.id, cardIds: masteredCardIds)
        }

        // Remove the card from remaining
        if currentIndex < remainingCards.count {
            remainingCards.remove(at: currentIndex)
        }
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
        restartSession()
    }

    func resetProgress() {
        guard let course = currentCourse else { return }
        masteredCardIds.removeAll()
        saveMasteredCards(for: course.id, cardIds: masteredCardIds)
        applyFilters()
    }
}
