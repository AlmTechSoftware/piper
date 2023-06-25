package main

import (
	"io"
	"log"
	"net"
	"sync"
)

func handleClient(conn net.Conn, wg *sync.WaitGroup, frameChan channel, resultChan channel) {
	log.Println("New client connected:", conn.RemoteAddr().String())
	defer func() {
		conn.Close()
		wg.Done()
		log.Println("Client disconnected:", conn.RemoteAddr().String())
	}()

	// Create a buffer to hold the incoming frame data
	buffer := make([]byte, 1024)

	for {
		// Read frame data from the client
		n, err := conn.Read(buffer)
		if err != nil {
			if err == io.EOF {
				return
			}
			log.Println("Error reading from client:", err)
			return
		}

		// Send to PiperWorker for processing
		frameData := buffer[:n]
		frameChan <- frameData
		log.Printf("Received frame from %s: %v\n", conn.RemoteAddr().String(), frameData)

		// Recieve data and return to client
		newFrameData := <-resultChan

		// Return new proc data to client
		_, err = conn.Write(newFrameData)
		if err != nil {
			log.Println("Error sending frame-data back to client:", err)
			return
		}
	}
}
