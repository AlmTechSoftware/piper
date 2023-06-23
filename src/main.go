package main

import (
	"fmt"
	"log"
	"net"
	"os"
	// "google.golang.org/grpc"
)

// type videoStreamServer struct {
// 	// Implement your gRPC service methods here
// }

func main() {
	var port = os.Getenv("PIPER_PORT")

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	fmt.Printf("lis: %v\n", lis)

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
