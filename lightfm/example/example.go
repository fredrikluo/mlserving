package main

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/fredrikluo/mlserving/lightfm"
	"github.com/gorilla/mux"
)

type lfmExample struct {
	router *mux.Router
	server *http.Server
	model  lightfm.Model
}

func newlfmExample() (*lfmExample, error) {
	return &(lfmExample{
		router: mux.NewRouter(),
		model:  lightfm.NewModel(),
	}), nil
}

func (lfmsrv *lfmExample) handleRecommendations(w http.ResponseWriter, r *http.Request) {
	statusCode := http.StatusOK
	response := map[string]interface{}{}
	defer lightfm.WriteResponse(&statusCode, &response, w)

	vars := mux.Vars(r)
	userID := vars["userId"]

	ret, err := lfmsrv.model.PredictFast(userID, 10)

	if err != nil {
		log.Printf("Error when calling Predict: %v", err)
		response["msg"] = err.Error()
		statusCode = http.StatusBadRequest
	}

	response["msg"] = ret
	log.Printf("Predict: %v", ret)
}

func (lfmsrv *lfmExample) init() {
	err := (&lfmsrv.model).Load("../movielens/model")
	if err != nil {
		log.Fatal(err)
	}

	lfmsrv.router.HandleFunc("/lfm/{userId}", lfmsrv.handleRecommendations).
		Methods("GET")

	lfmsrv.server = &http.Server{
		Handler:      lfmsrv.router,
		Addr:         "127.0.0.1:8080",
		WriteTimeout: 20 * time.Second,
		ReadTimeout:  20 * time.Second,
	}

	fmt.Printf("\nRunning at %s\n", lfmsrv.server.Addr)
}

func (lfmsrv *lfmExample) run() {
	lfmsrv.init()
	log.Fatal(lfmsrv.server.ListenAndServe())
}

func main() {
	lfmsrv, err := newlfmExample()
	if err != nil {
		log.Fatal(err)
	}

	lfmsrv.run()
}
