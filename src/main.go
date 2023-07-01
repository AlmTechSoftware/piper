package main

import (
	"log"
	"net"
	"os"
	"strconv"
	"sync"
	"syscall"
)

type channel = chan []byte

func initWorkers(numWorkers int) ([]*os.Process, []string, error) {
	workers := make([]*os.Process, numWorkers)
	socketPaths := make([]string, numWorkers)

	return workers, socketPaths, nil
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
	var numWorkers, err = strconv.Atoi(numWorkersStr)
	if err != nil {
		log.Fatalln("Unable to read max worker count from the environment!\nPlease set \"MAX_WORKER_COUNT\" to a valid positive integer.")
	}

	// Start a TCP server to accept client connections
	lis, err := net.Listen("tcp", ":"+port)

	if err != nil {
		log.Println("Error starting server:", err)
		return
	}

	defer lis.Close()


	var workers []*os.Process
	workers, socketPaths, err := initWorkers(numWorkers)
	if err != nil {
		return
	}

	defer func() {
		for _, wp := range workers {
			wp.Signal(syscall.SIGTERM)
			wp.Wait()
		}

		// Remove the worker socket files
		for _, path := range socketPaths {
			os.Remove(path)
		}
	}()

	log.Printf("Piper Server running on port %s", port)
	log.Println("Waiting for connections...")

	var wg sync.WaitGroup
	for {
		// FIXME: handle clients and create new workers
		// accept a new client connection
		conn, err := lis.Accept()
		if err != nil {
			log.Println("Error accepting client connection:", err)
			continue
		}

		// Handle the client connection asynchronously
		wg.Add(1)
		go handleClient(conn, &wg)
	}

	// NOTE: Will never reach but good to have I guess?
	wg.Wait()
	log.Println("Server shutting down...")
}
