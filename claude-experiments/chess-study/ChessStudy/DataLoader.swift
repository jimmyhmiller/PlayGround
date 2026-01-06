import Foundation

class DataLoader {
    let dataDir: String

    init(dataDir: String) {
        self.dataDir = dataDir
    }

    private func loadJSON<T: Codable>(from path: String) throws -> T {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }

    func loadCourses() throws -> [Course] {
        return try loadJSON(from: "\(dataDir)/courses.json")
    }

    func loadChapters(courseId: String) throws -> [Chapter] {
        return try loadJSON(from: "\(dataDir)/courses/\(courseId).json")
    }

    func loadStudies(chapterId: String) throws -> [Study] {
        return try loadJSON(from: "\(dataDir)/chapters/\(chapterId).json")
    }

    func loadVariationIds(studyId: String) throws -> [String] {
        return try loadJSON(from: "\(dataDir)/studies/\(studyId).json")
    }

    func loadVariation(studyId: String, variationId: String) throws -> Variation {
        return try loadJSON(from: "\(dataDir)/variations/\(studyId)/\(variationId).json")
    }

    func loadAllFlashcards(for course: Course, chapterIds: Set<String>? = nil) throws -> [Flashcard] {
        var allFlashcards: [Flashcard] = []

        let chapters = try loadChapters(courseId: course.id)
        let filteredChapters = chapterIds == nil ? chapters : chapters.filter { chapterIds!.contains($0.id) }

        for chapter in filteredChapters {
            let studies = try loadStudies(chapterId: chapter.id)

            for study in studies {
                let variationIds = try loadVariationIds(studyId: study.id)

                for varId in variationIds {
                    let variation = try loadVariation(studyId: study.id, variationId: varId)

                    let cards = generateFlashcards(
                        variation: variation,
                        playerColor: course.color,
                        courseName: course.name,
                        chapterId: chapter.id,
                        chapterName: chapter.name,
                        studyName: study.name
                    )
                    allFlashcards.append(contentsOf: cards)
                }
            }
        }

        // Deduplicate by stable ID
        var seen = Set<String>()
        return allFlashcards.filter { card in
            if seen.contains(card.id) {
                return false
            }
            seen.insert(card.id)
            return true
        }
    }
}
