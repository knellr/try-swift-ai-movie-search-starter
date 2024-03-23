import Foundation
import LargeLanguageModels
import SwiftData

class SearchManager: Logging  {
    
    struct SearchResult: Identifiable {
        let id: Int
        let movie: MovieItem
        let score: Double
    }
    
    let logger = PassthroughLogger()
    
    private let intelligence = AIIntelligenceManager.intelligence
    private var data: DataFrameIndex
    
    private let modelContext: ModelContext
    
    init(modelContainer: ModelContainer) throws {
        modelContext = ModelContext(modelContainer)
        data = try DataFrameIndex(MovieTextEmbeddingsIndexer.indexURL)
    }
    
    func search(_ text: String, maximumNumberOfResults: Int = 100) async throws -> [SearchResult] {
        let text = try await modifySearchQuery(text)
        
        logger.info("Searching with final query: \(text)")
        
        // get the embedding vector for the modified search text
        let searchEmbedding: [Double] = try await intelligence.textEmbedding(
            for: text,
            model: AIIntelligenceManager.embeddingModel
        ).rawValue
        
        // Query the movie text embeddings using the search embedding
        let embeddingSearchResults: [DataFrameIndex.SearchResult] = data.query(
            searchEmbedding,
            topK: maximumNumberOfResults
        )
        
        logger.info("Finished with \(embeddingSearchResults.count) result(s).")
        
        // Find the relevant movie id in SwiftData to match the result to the MovieItem in our dataset
        return try embeddingSearchResults.enumerated().map { (offset: Int, result: DataFrameIndex.SearchResult) in
            let id = UUID(uuidString: result.id)!
            let fetchDescriptor = FetchDescriptor(predicate: #Predicate<MovieItem> { movie in
                movie.id == id
            })
            
            let movieItem: MovieItem = try  modelContext.fetch(fetchDescriptor).first!
            
            return SearchResult(
                id: offset,
                movie: movieItem,
                score: result.score
            )
        }
    }
    
    private func modifySearchQuery(_ text: String) async throws -> String {
        let messages: [AbstractLLM.ChatMessage] = AIIntelligenceManager.messagesForText(text)
        
        let completion = try await intelligence.complete(
            prompt: AbstractLLM.ChatPrompt(messages: messages),
            model: AIIntelligenceManager.chatModel
        )
        let modifiedQuery = try String(completion.message.content)
        
        logger.info("Modified query:\n\"\(modifiedQuery)\"")
        
        return modifiedQuery
    }
}
