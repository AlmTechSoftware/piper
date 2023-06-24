package main

import (
	"log"
	"net"
	"os"
	"sync"
	// "google.golang.org/grpc"
)

// type videoStreamServer struct {
// 	// Implement your gRPC service methods here
// }

func main() {
	var port = os.Getenv("PIPER_PORT")
	if port == "" {
		port = "4242"
	}

	// Start a TCP server to accept client connections
	lis, err := net.Listen("tcp", ":"+port)

	if err != nil {
		log.Println("Error starting server:", err)
		return
	}

	defer lis.Close()

	log.Printf("Piper Server running on port %s", port)
	log.Println("Waiting for connections...")

	var wg sync.WaitGroup

	for {
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

	// // Create gRPC server
	// grpcServer := grpc.NewServer()
	//
	// // Register your gRPC service
	// videoStreamSvc := &videoStreamServer{}
	// RegisterVideoStreamServiceServer(grpcServer, videoStreamSvc)

	// // Start gRPC server
	// log.Println("Starting gRPC server...")
	// if err := grpcServer.Serve(lis); err != nil {
	// 	log.Fatalf("Failed to serve: %v", err)
	// }
}
