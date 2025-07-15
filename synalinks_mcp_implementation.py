import asyncio
import logging
import os
import json
import synalinks
import multiprocessing
import numpy as np
from typing import Dict, Any

# MCP and synalinks imports
from mcp.server.fastmcp import FastMCP
from synalinks.src.utils.mcp._test_common import run_streamable_server_multiprocessing
from synalinks.src.utils.mcp.client import MultiServerMCPClient
from synalinks.src.modules.agents.prebuilt import Agent

# Multirpocessing for MacOS
multiprocessing.set_start_method('fork')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Query(synalinks.DataModel):
    """Input query data model"""
    query: str = synalinks.Field(
        description="The user query",
    )

class FinalAnswer(synalinks.DataModel):
    """Final answer data model"""
    answer: str = synalinks.Field(
        description="The correct final answer",
    )


class MCPMathAgent:
    """MCP-based Math Agent with ReACT capabilities"""
    
    def __init__(self):
        self.status_server = None
        self.math_server = None
        self.status_server_context = None
        self.math_server_context = None
        self.client = None
        self.program = None
        
    def setup_servers(self):
        """Setup MCP servers with tools"""
        # Status server setup
        self.status_server = FastMCP(port=8182)
        
        @self.status_server.tool()
        def get_status() -> str:
            """Get server status"""
            return "Server is running"
            
        @self.status_server.tool()
        def get_uptime() -> str:
            """Get server uptime"""
            return "24h 30m 15s"
        
        # Math server setup
        self.math_server = FastMCP(port=8183)
        
        @self.math_server.tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together"""
            return a + b
            
        @self.math_server.tool()
        def multiply_numbers(a: int, b: int) -> int:
            """Multiply two numbers together"""
            return a * b
        @self.math_server.tool()
        def avg_numbers(a: int, b: int) -> int:
            """Average of two numbers"""
            return (a + b) // 2
            
        logger.info("MCP servers configured successfully")
    
    async def start_servers(self):
        """Start the MCP servers"""
        try:
            # Set up server contexts
            self.status_server_context = run_streamable_server_multiprocessing(self.status_server)
            self.math_server_context = run_streamable_server_multiprocessing(self.math_server)
            
            # Enter contexts
            self.status_server_context.__enter__()
            self.math_server_context.__enter__()
            
            logger.info("MCP servers started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            raise
    
    async def setup_client(self):
        """Setup MCP client with server connections"""

        status_connection = {
            "url": "http://localhost:8182/mcp/",
            "transport": "streamable_http",
        }
        
        math_connection = {
            "url": "http://localhost:8183/mcp/",
            "transport": "streamable_http",
        }
            
        try:
            self.client = MultiServerMCPClient({
                "status": status_connection,
                "math": math_connection,
            })
            
            logger.info("MCP client configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup client: {e}")
            raise
    
    async def create_agent(self):
        """Create the ReACT agent with MCP tools"""
        try:
            # Get all available tools from MCP servers
            all_tools = await self.client.get_tools()
            
            for tool in all_tools:
                tool._func.__name__ = tool._func.__name__.replace('/', '_')

            # Configure language model
            language_model = synalinks.LanguageModel(
                model="openai/gpt-4o-mini",
            )
            
            # Create input node
            x0 = synalinks.Input(data_model=Query)
            
            # Create prebuilt agent node
            x1 = await Agent(
                data_model=FinalAnswer,
                language_model=language_model,
                toolkit=[tool._func for tool in all_tools],
                max_iterations=10,
            )(x0)
            
            # Create program
            self.program = synalinks.Program(
                inputs=x0,
                outputs=x1,
                name="math_agent",
                description="A math agent that can use a calculator",
            )
            
            # Compile the program with reward and optimizer
            self.program.compile(
                reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
                optimizer=synalinks.optimizers.RandomFewShot(),
            )

            sample_query = Query(query="What is 2 + 2?")
            await self.program(sample_query)

            logger.info("ReACT agent created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    async def run_query(self, query: str) -> Dict[str, Any]:
        """Run a query through the math agent"""
        try:
            if not self.program:
                raise ValueError("Agent not initialized. Call setup() first.")
            
            # Create query input
            query_input = Query(query=query)
            
            # Run the program
            result = await self.program(query_input)
            
            logger.info(f"Query processed successfully: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            raise
    
    def load_datasets(self):
        """Load training, validation, and test datasets"""
        logger.info("Loading datasets...")
        
        # Load datasets from JSON file in the same folder
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        # Convert to DataModel objects (like GSM8k)
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        
        # Process training data
        for item in data["train"]:
            x_train.append(Query(**item["input"]))
            y_train.append(FinalAnswer(**item["output"]))
        
        # Process validation data  
        for item in data["validation"]:
            x_val.append(Query(**item["input"]))
            y_val.append(FinalAnswer(**item["output"]))
        
        # Process test data
        for item in data["test"]:
            x_test.append(Query(**item["input"]))
            y_test.append(FinalAnswer(**item["output"]))
        
        # Convert to numpy arrays with dtype="object" (like GSM8k)
        x_train = np.array(x_train, dtype="object")
        y_train = np.array(y_train, dtype="object")
        x_val = np.array(x_val, dtype="object")
        y_val = np.array(y_val, dtype="object")
        x_test = np.array(x_test, dtype="object")
        y_test = np.array(y_test, dtype="object")
        
        # Return in GSM8k format
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
            
    async def baseline_evaluation(self, test_dataset):
        """Perform baseline agent evaluation"""
        logger.info("Running baseline evaluation...")
        
        x_test, y_test = test_dataset
        nb_runs = getattr(self, 'nb_runs', 3)
        batch_size = getattr(self, 'batch_size', 32)
        
        try:
            baseline_metric_list = []
            
            for i in range(nb_runs):
                logger.info(f"Baseline run {i + 1}/{nb_runs}")
                
                # Evaluate the program
                metrics = await self.program.evaluate(
                    x=x_test,
                    y=y_test,
                    batch_size=batch_size,
                )
                
                baseline_metric_list.append(metrics)
                logger.info(f"Run {i + 1} metrics: {metrics}")
            
            # Calculate average metrics
            avg_metrics = self._calculate_average_metrics(baseline_metric_list)
            
            # Plot metrics if folder is specified
            if hasattr(self, 'folder'):
                synalinks.utils.plot_metrics_with_mean_and_std(
                    baseline_metric_list,
                    to_folder=self.folder,
                    title="Evaluation without training",
                )
            
            logger.info(f"Baseline evaluation completed. Average metrics: {avg_metrics}")
            return {
                "metrics_list": baseline_metric_list,
                "average_metrics": avg_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to perform baseline evaluation: {e}")
            raise



    
    async def optimize_agent(self, train_dataset, validation_dataset):
        """Optimize the agent using training data"""
        logger.info("Optimizing agent...")
        
        x_train, y_train = train_dataset
        x_val, y_val = validation_dataset
        
        nb_epochs = getattr(self, 'nb_epochs', 2)
        batch_size = getattr(self, 'batch_size', 32)
        folder = getattr(self, 'folder', "examples/training_programs")
        checkpoint_filepath = getattr(self, 'checkpoint_filepath', "checkpoint.program.json")
        
        try:
            # Setup checkpoint callback
            program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
                filepath=os.path.join(folder, checkpoint_filepath),
                monitor="val_reward",
                mode="max",
                save_best_only=True,
            )
            
            logger.info(f"Starting training for {nb_epochs} epochs...")
            
            # Train the program
            history = await self.program.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
                epochs=nb_epochs,
                batch_size=batch_size,
                callbacks=[program_checkpoint_callback],
            )
            
            # Plot training history
            synalinks.utils.plot_history(
                history,
                to_folder=folder,
                to_file="math_agent_training_history.png",
            )
            
            # Load best performing checkpoint
            logger.info("Loading best performing checkpoint...")
            self.program.load(os.path.join(folder, checkpoint_filepath))
            
            optimization_results = {
                "history": history,
                "epochs": nb_epochs,
                "best_checkpoint": os.path.join(folder, checkpoint_filepath),
                "convergence": True
            }
            
            logger.info("Agent optimization completed successfully")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize agent: {e}")
            raise

    
    async def optimized_evaluation(self, test_dataset):
    #     """Evaluate the optimized agent"""
        logger.info("Running optimized agent evaluation...")
        
        x_test, y_test = test_dataset
        nb_runs = getattr(self, 'nb_runs', 3)
        batch_size = getattr(self, 'batch_size', 32)
        
        try:
            optimized_metric_list = []
            
            for i in range(nb_runs):
                logger.info(f"Optimized run {i + 1}/{nb_runs}")
                
                # Evaluate the trained program
                metrics = await self.program.evaluate(
                    x=x_test,
                    y=y_test,
                    batch_size=batch_size,
                )
                
                optimized_metric_list.append(metrics)
                logger.info(f"Run {i + 1} metrics: {metrics}")
            
            # Calculate average metrics
            avg_metrics = self._calculate_average_metrics(optimized_metric_list)
            
            logger.info(f"Optimized evaluation completed. Average metrics: {avg_metrics}")
            return {
                "metrics_list": optimized_metric_list,
                "average_metrics": avg_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to perform optimized evaluation: {e}")
            raise
    def _calculate_average_metrics(self, metrics_list):
        """Calculate average metrics from a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        # Get all metric keys from the first entry
        keys = metrics_list[0].keys()
        avg_metrics = {}
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def plot_comparison(self, baseline_results, optimized_results):
        """Plot comparison between baseline and optimized results"""
        try:
            folder = getattr(self, 'folder', "examples/training_programs")
            
            metrics_comparison = {
                "without_training": baseline_results["metrics_list"],
                "with_training": optimized_results["metrics_list"],
            }
            
            synalinks.utils.plot_metrics_comparison_with_mean_and_std(
                metrics_comparison,
                to_folder=folder,
                to_file="math_agent_evaluation_comparison.png",
                title="Comparison w/o training (Math Agent with EM reward)",
            )
            
            logger.info("Comparison plot saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to plot comparison: {e}")
    def set_training_params(self, nb_epochs=2, batch_size=32, nb_samples=None, nb_runs=3, folder="examples/training_programs"):
        """Set training parameters"""
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.nb_runs = nb_runs
        self.folder = folder
        self.checkpoint_filepath = "checkpoint.program.json"
        
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
    
    async def setup(self):
        """Complete setup process"""
        self.setup_servers()
        await self.start_servers()
        await self.setup_client()
        await self.create_agent()
        logger.info("Math agent setup completed successfully")
    
    def cleanup(self):
        """Clean up server contexts"""
        try:
            if self.status_server_context:
                self.status_server_context.__exit__(None, None, None)
                logger.info("Status server context cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up status server context: {e}")
        
        try:
            if self.math_server_context:
                self.math_server_context.__exit__(None, None, None)
                logger.info("Math server context cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up math server context: {e}")

async def main():
    """Main execution function"""
    agent = MCPMathAgent()
    
    try:
        # Setup the agent
        await agent.setup()
        
        agent.set_training_params(
            nb_runs=1,
            nb_epochs=1,
            batch_size=4,
        )

        # Load datasets
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = agent.load_datasets()
        
        # Create dataset tuples
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)
        
        # Baseline evaluation
        baseline_results = await agent.baseline_evaluation(test_data)
        logger.info(f"Baseline results: {baseline_results}")
        
        # Agent optimization
        optimization_results = await agent.optimize_agent(train_data, val_data)
        logger.info(f"Optimization results: {optimization_results}")
        
        # Optimized evaluation
        optimized_results = await agent.optimized_evaluation(test_data)
        logger.info(f"Optimized results: {optimized_results}")
        
        # Example query
        example_query = "What is 25 + 17 multiplied by 3?"
        result = await agent.run_query(example_query)
        logger.info(f"Query result: {result}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
        
    finally:
        # Cleanup
        agent.cleanup()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
