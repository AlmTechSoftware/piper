package main

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
)

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

func handleClient(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("Failed to upgrade connection", err)
		return
	}

	// TODO: implement a queue
	// TODO: check if max workers reached, if so then put into queue

	go processClient(conn, workers)
}

func processClient(conn *websocket.Conn, workers []*PiperWorker) {
	log.Println("New client connected:", conn.RemoteAddr().String())

	// Create piperworker
	worker := newPiperWorker()
	defer func() {
		conn.Close()
		worker.kill()
	}()

	// Append worker to workers
	workers = append(workers, worker)

	for {
		_, frameData, err := conn.ReadMessage()
		if err != nil {
			log.Println("Failed to read WS message:", err)
			return
		}

		// TODO: async

		// Send the frame data to the worker
		worker.send(frameData)

		// Now recv data from the worker and send back to client
		frameData, err = worker.recv(BUFFER_SIZE)
		if err != nil {
			log.Println("Frame data processing failed:", err)
			return
		}

		// Returned the processed frames to the client
		err = conn.WriteMessage(1, frameData)
		if err != nil {
			log.Println("Unable to send data back to client:", err)
			return
		}
	}
}
