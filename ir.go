/* 
Package go-ir provide a information retrieval engine. 

The engine computes cosine similarities between its documents and a given text query using the tf-idf metric.

Example

One example ranking html documents by relevance to given query:

    eng = ir.NewEngine()
    for doc := html_documents {
        eng.AddDocument(doc.Url, doc.Html)
    }
    eng.Vectorize()
    query_results := eng.Query("keyword")

    for _, result := range query_results {
        fmt.Println(result.Id, result.Score)
    }
*/
package ir

import (
    "regexp"
    "strings"
    "math"
    "encoding/json"
)

// ======================== Types declarations =======================================

// A Document is composed of an Id and a map Tfidf sending each word to its tf-idf score in the document.
// Its vocabulary can be accessed via the keys in the Tfidf map. 
type Document struct {
    Id string                   `json:id"`      //Can be any identifier
    Tfidf map[string] float64   `json:"tfidf"`  // Calculate once, use in each search
}

// Each Engine ontains an array of Document and a map Idf. 
// Its vocabulary can be accessed via the keys in the Idf map. 
type Engine struct {
    Documents []Document    `json:"documents"`
    Idf map[string] float64 `json:"idf"`    // Calculated once, used in each query
}

// Used to store the cosine of the angle between the document identified by Id and the query.
type SearchResult struct {
    Id string       `json:"id"`
    Score float64   `json:"score"`
}

// ======================== Engine methods ===========================================

// Create a new Engine struct.
func NewEngine() *Engine {
    eng := new(Engine)
    eng.Documents = make([]Document, 0)
    eng.Idf = make(map[string] float64)
    return eng
}

// Add new document to the Engine.
// The document tf-idf is initialized with simple term frequency.
// Indeed, we need all documents to compute idf and tf-idf.
// This computation is done with Vectorize().
func (eng *Engine) AddDocument(id string, body string) {
    doc := new(Document)
    doc.Id = id
    doc.Tfidf = Tf(body) 

    eng.Documents = append(eng.Documents, *doc)
}

// Vectorize the Documents in the Engine.
// This function will populate the maps Idf and Tfidf.
func (eng *Engine) Vectorize() {

    // Compute Document Frequency (df) for each word
    df := make(map[string] int)
    for _, doc := range eng.Documents {
        for word, _ := range doc.Tfidf {
            df[word] = df[word] + 1 
        }
    }

    // Compute Inverse Document Frequency (idf) for each word
    vocabulary_size := float64(len(df))
    for word, _ := range df {
            eng.Idf[word] = math.Log( vocabulary_size/( 1 + float64(df[word])))
    }

    // Compute tf-idf for each word relative to each document (like a sparse matrix) and normalize.
    for _, doc := range eng.Documents {
        squared_norm := float64(0)
        for word, tf := range doc.Tfidf {
            doc.Tfidf[word] = tf * eng.Idf[word]
            squared_norm = squared_norm +  doc.Tfidf[word] * doc.Tfidf[word]
        }
        // Normalize tfidf row (for one document)
        norm := math.Sqrt(squared_norm)
        for word, tfidf := range doc.Tfidf {
            doc.Tfidf[word] = tfidf / norm
         }
    }
}

// Make a query against the Engine documents.
// It returns an ordered (by score) array of SearchResult. 
// Only score > 0 are returned.
func (eng *Engine) Query(text string) []SearchResult {
    query_vec :=Tf(text)

    // Compute query vector for given search text.
    squared_norm := float64(0)
    for word, tf := range query_vec {
        query_vec[word] = tf * eng.Idf[word] // That's why we pre-computed Idf
        squared_norm = squared_norm +  query_vec[word] * query_vec[word]
    }

    // Normalize query vector.
    norm := math.Sqrt(squared_norm)
    for word, tfidf := range query_vec {
        query_vec[word] = tfidf / norm
    }

    // Compute scalar products between the query vector and each document. 
    results := make([]SearchResult, 0)
    for _, doc := range eng.Documents {
        scalar_product := float64(0)
        for word, _ := range query_vec {
                scalar_product = scalar_product + query_vec[word] * doc.Tfidf[word]
            }
        if(scalar_product > 0) {
            results = append(results, SearchResult{doc.Id, scalar_product}) 
        }
    }

    // Sort results by score (scalar procuct, cosine similarity).
    decreasing_score := func(r1, r2 *SearchResult) bool {
        return r1.Score > r2.Score
    }
    By(decreasing_score).Sort(results)
    
    return results
}

// Return a JSON object for the engine.
func (eng *Engine) Json() []byte {
    b,_ := json.MarshalIndent(eng, "", "  ")
    return b
} 

// ======================== Auxiliary functions ======================================

/* Pre-process given text following this steps: 
    * Remove all html tags.
    * Remove all non-words, including punctuation.
    * Remove whitespaces.
*/
func Preprocess(text string) string {
        r1 := regexp.MustCompile("<[^<>]+>") // Remove HTML tags
        r2 :=  regexp.MustCompile("[^a-z\\xE0\\xE1\\xE2\\xE3\\xE4\\xE5\\xE6\\xE7\\xE8\\xE9\\xEA\\xEB\\xEC\\xED\\xEE\\xEF\\xF0\\xF1\\xF2\\xF3\\xF4\\xF5\\xF6\\xF7\\xF8\\xF9\\xFA\\xFB\\xFC\\xFD\\xFE\\xFF]") // Leave only words for tokenization (numbers following words: terminals models)
        r3 :=  regexp.MustCompile("[\\n\\r\\s]+") // Remove whitespaces
        return strings.Trim(  r3.ReplaceAllString( r2.ReplaceAllString( r1.ReplaceAllString(strings.ToLower(text), " "), " " ), " "), " ")
}

// Return a map with weighted term frequencies for given text.
func Tf(text string) map[string] float64 {
    text = Preprocess(text)

    f := make(map[string] int) // Raw term frequence (f(word, document))
    tf := make(map[string] float64) // Weighted term frequence (tf(word, document))

    for _, word := range strings.Split(text, " ") {
        f[word] = f[word] + 1
    }
    
    for word, count := range f {
            tf[word] = math.Log(1 + float64(count)) 
    }

    return tf
}
