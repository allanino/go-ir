package ir

// ======================= Functions implementing a sort.Iterface ======================

// By is the type of a "less" function that defines the ordering.
type By func(r1, r2 *SearchResult) bool

// Sort function for ordering the search results.
func (by By) Sort(results []SearchResult) {
    rs := &resultsSorter{
        results: results,
        by:      by, // The Sort method's receiver is the function (closure) that defines the sort order.
    }
    sort.Sort(rs)
}

//  Joins a By function and a slice of SearchResult to be sorted.
type resultsSorter struct {
    results []SearchResult
    by func(r1, r2 *SearchResult) bool // Closure used in the Less method.
}

// Len is part of sort.Interface.
func (s *resultsSorter) Len() int {
    return len(s.results)
}

// Swap is part of sort.Interface.
func (s *resultsSorter) Swap(i, j int) {
    s.results[i], s.results[j] = s.results[j],s.results[i]
}

// Less is part of sort.Interface. It is implemented by calling the "by" closure in the sorter.
func (s *resultsSorter) Less(i, j int) bool {
    return s.by(&s.results[i], &s.results[j])
}