package main

import (
	"log"
	"net"

	"google.golang.org/grpc"
)

type videoStreamServer struct {
	// Implement your gRPC service methods here
}

func main() {
	// gRPC server initialization
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	// Create gRPC server
	grpcServer := grpc.NewServer()

	// Register your gRPC service
	videoStreamSvc := &videoStreamServer{}
	RegisterVideoStreamServiceServer(grpcServer, videoStreamSvc)

	// Start gRPC server
	log.Println("Starting gRPC server...")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
