import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, List


class LoadBalancer:
    def __init__(self, input_pipe: mp.Queue, num_workers: int = 8) -> None:
        self.input_pipe = input_pipe
        self.num_workers = num_workers
        self.output_pipes = []

        self.workers = []
        self.tasks = []

        # Create multiple pipes for communication with the worker processes
        for _ in range(self.num_workers):
            self.output_pipes.append(mp.Pipe())

    def start_workers(self) -> None:
        for i in range(self.num_workers):
            # Create a worker process, passing the input pipe and output pipe as arguments
            worker_proc = mp.Process(
                target=self.worker, args=(self.input_pipe, self.output_pipes[i][1])
            )
            worker_proc.start()  # Start the worker process
            self.workers.append(worker_proc)  # Add the worker process to the list

    def worker(self, input_pipe: mp.Queue, output_pipe: Connection) -> None:
        while True:
            task = input_pipe.recv()  # Receive a task from the input pipe
            if (
                task is None
            ):  # If task is None, it's a signal to exit the loop and terminate the worker process
                break
            # Process the task
            result = self.process_task(
                task
            )  # Call the process_task function to perform the computation on the task
            output_pipe.send(result)  # Send the result through the output pipe

    def process_task(self, task: Any) -> Any:
        # Process the task and return the result
        return task  # TODO: push into Piper algo

    def collect_results(self) -> List[Any]:
        results = []  # List to store the results

        # Collect results from the worker processes
        for i in range(self.num_workers):
            result = self.output_pipes[i][
                0
            ].recv()  # Receive the result from the output pipe
            results.append(result)  # Add the result to the results list

        return results

    def wait_for_workers(self) -> None:
        # Wait for all worker processes to finish
        for worker_process in self.workers:
            worker_process.join()  # Wait for each worker process to terminate

    def process_results(self, results: List[Any]) -> None:
        # Process the results
        for result in results:
            self.__process_result(
                result
            )  # Call the process_result function to handle each result

    def __process_result(self, result: Any) -> None:
        # Placeholder function for handling the results
        print("Result:", result)
