package main

import (
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/gorilla/websocket"
)

type channel = chan []byte

func initWorkers(numWorkers int) []*PiperWorker {
	workers := make([]*PiperWorker, numWorkers)

	return workers
}

var BUFFER_SIZE = 1024

var upgrader = websocket.Upgrader{
	ReadBufferSize:  BUFFER_SIZE,
	WriteBufferSize: BUFFER_SIZE,
}

var maxNumWorkers int = 0
var workers []*PiperWorker

func handleClient(conn *websocket.Conn) {
	defer conn.Close()

	// TODO: implement a queue
	// TODO: check if max workers reached, if so then put into queue

	handlePiper(conn, workers)
}

func main() {
	// Parse .env stuff

	// PORT
	var port = os.Getenv("PORT")
	if port == "" {
		port = "4242"
	}

	// WORKER COUNT
	var numWorkersStr = os.Getenv("MAX_WORKER_COUNT")
	maxNumWorkers, err := strconv.Atoi(numWorkersStr)
	if err != nil {
		log.Fatalln("Unable to read max worker count from the environment!\nPlease set \"MAX_WORKER_COUNT\" to a valid positive integer.")
	}

	// Create worker container
	workers = initWorkers(maxNumWorkers)
	defer func() {
		for _, wp := range workers {
			wp.kill()
		}
	}()

	log.Printf("Piper Server running on port %s", port)
	log.Println("Waiting for connections...")

	http.HandleFunc("/piper", handleClient)
	err = http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatalln("Websocket server failed:", err)
	}

	log.Println("Server shutting down...")
}
