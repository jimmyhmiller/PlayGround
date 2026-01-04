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

    func loadAllFlashcards(for course: Course) throws -> [Flashcard] {
        var allFlashcards: [Flashcard] = []

        let chapters = try loadChapters(courseId: course.id)

        for chapter in chapters {
            let studies = try loadStudies(chapterId: chapter.id)

            for study in studies {
                let variationIds = try loadVariationIds(studyId: study.id)

                for varId in variationIds {
                    let variation = try loadVariation(studyId: study.id, variationId: varId)

                    let cards = generateFlashcards(
                        variation: variation,
                        playerColor: course.color,
                        courseName: course.name,
                        chapterName: chapter.name,
                        studyName: study.name
                    )
                    allFlashcards.append(contentsOf: cards)
                }
            }
        }

        // Deduplicate
        var seen = Set<String>()
        return allFlashcards.filter { card in
            let key = "\(card.positionFen)|\(card.correctMove)"
            if seen.contains(key) {
                return false
            }
            seen.insert(key)
            return true
        }
    }
}
