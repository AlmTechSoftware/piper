package main

import (
	"fmt"
	"io"
	"net"
	"sync"
)

func handleClient(conn net.Conn, wg *sync.WaitGroup) {
	fmt.Println("New client connected:", conn.RemoteAddr().String())
	defer func() {
		conn.Close()
		wg.Done()
		fmt.Println("Client disconnected:", conn.RemoteAddr().String())
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
			fmt.Println("Error reading from client:", err)
			return
		}

		// Process the received frame data here
		frameData := buffer[:n]
		fmt.Printf("Received frame from %s: %v\n", conn.RemoteAddr().String(), frameData)
	}
}
