package embeddings

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetNnsByItem(t *testing.T) {
	annIndex, err := NewAnnIndex("testdata/test.ann", "testdata/index2id.json", 2)
	assert.Nil(t, err)

	result := annIndex.GetNnsByItem(0, 3, -1)
	assert.Equal(t, []int{0, 1, 2}, result)
	id, _ := annIndex.IndexToId(result[0])
	assert.Equal(t, "0", id)
}

func TestGetNnsByVector(t *testing.T) {
	annIndex, err := NewAnnIndex("testdata/test.ann", "testdata/index2id.json", 2)
	assert.Nil(t, err)

	result := annIndex.GetNnsByVector([]float32{0.8944271909999159, 0.4472135954999579}, 3, -1)
	assert.Equal(t, []int{2, 0, 1}, result)
}
