import Foundation
import SwiftData
import OrderedCollections

/// This represents a searchable index of movies in the app.
///
/// - Indexing the movies using OpenAI's text embeddings
class MovieTextEmbeddingsIndexer {
    
    private let modelContext: ModelContext
    
    private let intelligence = AIIntelligenceManager.intelligence
    
    // Save DataFrame table into a new CSV file, which can be loaded and searched later
    static let indexURL = URL.documentsDirectory.appendingPathComponent("textEmbeddingsIndex.csv")
    
    
    private var data: DataFrameIndex
    
    init(modelContainer: ModelContainer) throws {
        self.modelContext = ModelContext(modelContainer)
        
        data = try DataFrameIndex(MovieTextEmbeddingsIndexer.indexURL)
    }
    
    /// Indexes all the movies.
    ///
    /// Resets the index each time this is run.
    func indexAll() async throws {
        let movies = try modelContext.fetch(FetchDescriptor<MovieItem>())
        
        /// This assumes that the dataset is static (i.e. if the number of movies are the same, it means all movies have been added to the index).
        guard data.keys.count != movies.count else {
            return
        }
        
        // reset the data
        data = DataFrameIndex()
        
        let descriptionsByID: OrderedCollections.OrderedDictionary<UUID, String> = OrderedDictionary(
            uniqueKeysWithValues: movies.map(
                { ($0.id, // movie UUID
                   AIIntelligenceManager.descriptionForMovie($0))
                })
        )
        
        // Get the vector values of the movie descriptions from OpenAI's text embeddings API
        let embeddings = try await intelligence.textEmbeddings(
            for: Array(descriptionsByID.values),
            model: AIIntelligenceManager.embeddingModel
        )
        
        let embeddingsWithIDs = embeddings.enumerated().map({
            (
                descriptionsByID.elements[$0.offset].key.stringValue,
                $0.element.embedding.rawValue
            )
        })
        
        // Add the mappings of the movie UUID + description vectors to the DataFrame and save it to a CSV file
        data.insert(contentsOf: embeddingsWithIDs)
        
        try data.save(to: MovieTextEmbeddingsIndexer.indexURL)
    }
}
