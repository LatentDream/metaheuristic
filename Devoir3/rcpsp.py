import networkx as nx
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RCPSP:
    def __init__(self, filename: str):
        """Initialize the RCPSP structure from input file. The resulting graph will
        have nodes with the following attributes:
        - index (int): the index of the node. INDEXES start at 1
        The graph will have edges with the following attributes:
        - weight (float): weight of the edge
        Args:
            filename (str): path to the instance file
        """
        jobs, durations, resources, resource_availabilities = self.parse_psplib_data(
            filename
        )
        self.resource_availabilities = resource_availabilities
        self.graph = self.create_graph(jobs, durations, resources)

    def parse_psplib_data(self, filename: str):
        """Parse the given instance and store the result as a networkx graph.

        Args:
            filename (str): path to the instance to parse

        Returns:
            jobs (list): id of jobs
            durations (dict): maps job IDs to their duration
            resources (dict): maps job IDs to their resource consumption; ex {1: [2, 3, 5, 0]}
            resource_availabilities (list[int]): the availability of all resources
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        jobs_data_index = (
            lines.index("PRECEDENCE RELATIONS:\n") + 2
        )  # Skip the header line
        jobs_data = lines[jobs_data_index:]

        jobs = {}
        for line in jobs_data:
            if line.startswith(
                "************************************************************************"
            ):
                break

            job_info = list(map(int, line.split()))
            job_id, _, *successors = job_info
            jobs[job_id] = successors

        durations_index = (
            lines.index("REQUESTS/DURATIONS:\n") + 2
        )  # Skip the header line
        durations_data = lines[durations_index:]

        durations = {}
        resources = {}
        for line in durations_data:
            if line.startswith(
                "------------------------------------------------------------------------"
            ) or line.startswith(
                "************************************************************************"
            ):
                continue
            if line.startswith("RESOURCEAVAILABILITIES:"):
                break

            job_info = list(map(int, line.split()))
            job_id, duration, *resource_reqs = job_info
            durations[job_id] = duration
            resources[job_id] = resource_reqs

        resource_availabilities = [int(i) for i in lines[-2].split()]

        return jobs, durations, resources, resource_availabilities

    def create_graph(self, jobs, durations, resources):
        """Builds the graph for the RCPSP problem

        The node IDs are the task (job) IDs, and all nodes have attributes duration and resources
        """
        graph = nx.DiGraph()

        for job_id, successors in jobs.items():
            graph.add_node(
                job_id, duration=durations[job_id], resources=resources[job_id]
            )

            for successor in successors:
                graph.add_edge(job_id, successor)

        return graph

    def display_solution(self, solution: Dict, plot_name: str = ""):
        """Displays a solution with 2 plots:
        -a gantt chart that represents the relations between predecessors and successors
        -a plot indicating the current ressource utilization at all time points
        Args:
            solution (dict): A dictionary mapping task_ids to their start time.
            plot_name (str): name of plots
        """
        self.plot_resource_utilization(solution, plot_name)
        self.create_gantt_chart(solution, plot_name)

    def create_gantt_chart(self, solution, plot_name=""):
        """
        Create a Gantt chart with arrows connecting tasks based on their successors in the graph.

        :param solution: A dictionary mapping task_ids to their start time.
        :param filename: The filename to save the chart as an image. Leave blank to not save the image.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for task_id, start_time in solution.items():
            duration = self.graph.nodes[task_id]["duration"]
            end_time = start_time + duration
            ax.barh(
                task_id, duration, left=start_time, align="center", color="lightblue"
            )
            ax.text(
                start_time + (duration / 2),
                task_id,
                f"Task {task_id}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

        # Add arrows for dependencies
        for task_id, task_start in solution.items():
            task_end = task_start + self.graph.nodes[task_id]["duration"]
            successors = self.graph.successors(task_id)
            for succ_id in successors:
                succ_start = solution[succ_id]
                arrow = mpatches.FancyArrowPatch(
                    (task_end, task_id),
                    (succ_start, succ_id),
                    arrowstyle="-|>",
                    mutation_scale=20,
                    color="red",
                    linestyle="dashed",
                )
                ax.add_patch(arrow)

        max_finish_time = max(
            [
                solution[job] + self.graph.nodes[job]["duration"]
                for job in self.graph.nodes
            ]
        )
        ax.set_xlim(0, max_finish_time)
        ax.set_xlabel("Time")
        ax.set_ylabel("Task ID")
        ax.set_title("Gantt Chart")
        ax.grid(True)

        if plot_name:
            plt.savefig(plot_name + "_gantt.png", bbox_inches="tight")
        plt.show()

    def plot_resource_utilization(self, solution, plot_name=""):
        """
        Create a bar chart for ressource utilization at all time points

        solution (dict): A dictionary mapping task_ids to their start time.
        plot_name (str): name of the plot
        """
        num_resources = len(self.resource_availabilities)
        max_finish_time = max(
            [
                solution[job] + self.graph.nodes[job]["duration"]
                for job in self.graph.nodes
            ]
        )

        # Initialize resource utilization
        resource_utilization = {
            i: [0] * (max_finish_time + 1) for i in range(num_resources)
        }

        # Extract the unique start and finish times and sort them
        time_points = sorted(
            set(
                [solution[job] for job in self.graph.nodes]
                + [
                    solution[job] + self.graph.nodes[job]["duration"]
                    for job in self.graph.nodes
                ]
            )
        )

        for idx in range(len(time_points) - 1):
            t_start, t_end = time_points[idx], time_points[idx + 1]

            for job, job_start in solution.items():
                job_duration = self.graph.nodes[job]["duration"]
                job_finish = job_start + job_duration

                if job_start <= t_start < job_finish:
                    job_resources = nx.get_node_attributes(self.graph, "resources")[job]
                    for i, r in enumerate(job_resources):
                        resource_utilization[i][t_start:t_end] = [
                            x + r for x in resource_utilization[i][t_start:t_end]
                        ]

        # Create subplots for each resource
        fig, axes = plt.subplots(
            num_resources, 1, figsize=(12, num_resources * 4), sharex=True
        )

        for i, (ax, resource_usage) in enumerate(
            zip(axes, resource_utilization.values())
        ):
            ax.bar(range(max_finish_time + 1), resource_usage)
            ax.axhline(
                y=self.resource_availabilities[i],
                color="r",
                linestyle="-",
                label=f"Capacity: {self.resource_availabilities[i]}",
            )
            ax.set_title(f"Resource {i + 1}")
            ax.legend()

        plt.xlabel("Time")
        plt.tight_layout()
        if plot_name:
            plt.savefig(plot_name + "_resources.png", bbox_inches="tight")
        plt.show()

    def verify_solution(self, solution):
        # min start time is 0
        min_start_time = min([solution[job] for job in self.graph.nodes])
        if min_start_time != 0:
            print(f"Start time is not 0")
            return False
        # Check precedence constraints
        for job in self.graph.nodes:
            duration = self.graph.nodes[job]["duration"]
            job_start_time = solution[job]
            job_finish_time = job_start_time + duration
            for successor in self.graph.successors(job):
                if solution[successor] < job_finish_time:
                    print(
                        f"Precedence constraint violated: job {job} -> job {successor}"
                    )
                    return False

        # Check resource constraints
        num_resources = len(self.resource_availabilities)

        # Find the maximum finish time to set the range for resource usage check
        max_finish_time = max(
            [
                solution[job] + self.graph.nodes[job]["duration"]
                for job in self.graph.nodes
            ]
        )

        for t in range(max_finish_time + 1):
            resource_usage = [0] * num_resources
            for job, start_time in solution.items():
                job_finish_time = start_time + self.graph.nodes[job]["duration"]
                if start_time <= t < job_finish_time:  # Fix the condition here
                    job_resources = nx.get_node_attributes(self.graph, "resources")[job]
                    resource_usage = [
                        x + y for x, y in zip(resource_usage, job_resources)
                    ]

            if any(
                usage > available
                for usage, available in zip(
                    resource_usage, self.resource_availabilities
                )
            ):
                print(
                    f"Resource constraint violated at time {t}: {resource_usage} > {self.resource_availabilities}"
                )
                return False

        return True

    def get_solution_cost(self, solution: Dict) -> int:
        """Computes and returns the cost of a solution
        Args:
            solution (Dict): maps task ID to start time
        Returns:
            max_finish_time (int): cost (makespan) of the solution
        """
        max_finish_time = max(
            [
                solution[job] + self.graph.nodes[job]["duration"]
                for job in solution.keys()
            ]
        )

        return max_finish_time

    def save_solution(self, solution: Dict, output_file: str = "") -> None:
        """Saves solution to file

        Args:
            solution (Dict): maps task ID to start time
            output_file (str, optional): name of the output file where the solution will be saved
        """
        with open(output_file, "w") as file:
            for k, v in solution.items():
                file.write(str(k) + "," + str(v) + "\n")
