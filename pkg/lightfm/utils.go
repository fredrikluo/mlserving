package lightfm

import (
	"encoding/json"
	"log"
	"net/http"
)

//WriteResponse write generic response
func WriteResponse(code *int, response interface{}, w http.ResponseWriter) {
	jsonData, err := json.Marshal(response)
	if err != nil {
		log.Fatal(err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(*code)
	w.Write(jsonData)
}
