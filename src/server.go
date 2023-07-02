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

	go processClient(conn, workers)
}

func processClient(conn *websocket.Conn, workers []*PiperWorker) {
	log.Println("New client connected:", conn.RemoteAddr().String())

	// Create piperworker
	worker := newPiperWorker()
	// defer func() {
	// 	conn.Close()
	// 	worker.kill()
	// }()

	// Append worker to workers
	workers = append(workers, worker)

	// Channel to receive processed frame data
	processedDataChan := make(chan []byte)

	// Goroutine to handle receiving and sending back data
	go func() {
		for {
			_, frameData, err := conn.ReadMessage()
			if err != nil {
				// Connection closed / connection error
				log.Println("Connection error:", err)
				return
			}

			// Perform processing of frame data
			go worker.send(frameData)

			// Asynchronously wait for the processed data to be done
			go func(frameData []byte) {
				processedData, err := worker.recv(BUFFER_SIZE)
				if err != nil {
					log.Println("Frame data processing failed:", err)
					return
				}
				processedDataChan <- processedData
			}(frameData)

			// Returned the processed frames to the client
			select {
			case processedData := <-processedDataChan:
				err = conn.WriteMessage(websocket.BinaryMessage, processedData)
				if err != nil {
					log.Println("Unable to send data back to client:", err)
					return
				}
			}
		}
	}()
}
