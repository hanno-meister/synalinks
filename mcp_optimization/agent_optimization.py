import synalinks
import logging
import os
import json
import numpy as np
import asyncio

from mcp_checkpoint import MCPProgramCheckpoint

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
        self.client = None
        self.program = None

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
            self.client = synalinks.MultiServerMCPClient({
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
            all_tools = await self.client.get_tools()
            
            for tool in all_tools:
                tool._func.__name__ = tool._func.__name__.replace('/', '_')

            language_model = synalinks.LanguageModel(model="openai/gpt-4.1-nano")
            embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-ada-002")
            
            x0 = synalinks.Input(data_model=Query)
            x1 = await synalinks.FunctionCallingAgent(
                data_model=FinalAnswer,
                language_model=language_model,
                tools=all_tools,
                max_iterations=3,
            )(x0)
            
            self.program = synalinks.Program(
                inputs=x0,
                outputs=x1,
                name="math_agent",
                description="A math agent that can use a calculator",
            )
            
            self.program.compile(
                reward=synalinks.rewards.CosineSimilarity(embedding_model=embedding_model, in_mask=["answer"]),
                optimizer=synalinks.optimizers.RandomFewShot(),
            )

            sample_query = Query(query="What is 2 + 2?")
            await self.program(sample_query)
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    async def run_query(self, query: str):
        """Run a query through the math agent"""
        try:
            if not self.program:
                raise ValueError("Agent not initialized. Call setup() first.")
            
            query_input = Query(query=query)
            result = await self.program(query_input)
                
            logger.info(f"Query processed successfully: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            raise
    
    def load_datasets(self):
        """Load training, validation, and test datasets"""
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        
        for item in data["train"]:
            x_train.append(Query(**item["input"]))
            y_train.append(FinalAnswer(**item["output"]))
        
        for item in data["validation"]:
            x_val.append(Query(**item["input"]))
            y_val.append(FinalAnswer(**item["output"]))
        
        for item in data["test"]:
            x_test.append(Query(**item["input"]))
            y_test.append(FinalAnswer(**item["output"]))
        
        x_train = np.array(x_train, dtype="object")
        y_train = np.array(y_train, dtype="object")
        x_val = np.array(x_val, dtype="object")
        y_val = np.array(y_val, dtype="object")
        x_test = np.array(x_test, dtype="object")
        y_test = np.array(y_test, dtype="object")
        
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
                
                metrics = await self.program.evaluate(
                    x=x_test,
                    y=y_test,
                    batch_size=batch_size,
                )
                
                baseline_metric_list.append(metrics)
                logger.info(f"Run {i + 1} metrics: {metrics}")
            
            avg_metrics = self._calculate_average_metrics(baseline_metric_list)
            
            logger.info(f"Baseline evaluation completed. Average metrics: {avg_metrics}")
            return {
                "metrics_list": baseline_metric_list,
                "average_metrics": avg_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to perform baseline evaluation: {e}")
            raise

    async def optimize_agent(self, train_dataset, validation_dataset):
        """Optimize the agent using training data with MCP-safe checkpointing"""
        x_train, y_train = train_dataset
        x_val, y_val = validation_dataset
        
        nb_epochs = getattr(self, 'nb_epochs', 2)
        batch_size = getattr(self, 'batch_size', 32)
        folder = getattr(self, 'folder', "examples/training_programs")
        checkpoint_filepath = "mcp_checkpoint.json"
        
        try:
            
            mcp_checkpoint = MCPProgramCheckpoint(
                filepath=os.path.join(folder, checkpoint_filepath),
                monitor="val_reward",
                mode="max",
                save_best_only=True,
                verbose=1
            )
            
            history = await self.program.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
                epochs=nb_epochs,
                batch_size=batch_size,
                callbacks=[mcp_checkpoint],
            )
            
            # Load best checkpoint
            logger.info("Loading best checkpoint...")
            mcp_checkpoint.load_program_state(self.program, self.client)
            
            return {"training_completed": True, "best_metric": mcp_checkpoint.best}
            
        except Exception as e:
            logger.error(f"Failed to optimize agent: {e}")
            return {"training_completed": False, "error": str(e)}

    async def optimized_evaluation(self, test_dataset):
        """Evaluate the optimized agent"""
        logger.info("Running optimized agent evaluation...")
        
        x_test, y_test = test_dataset
        nb_runs = getattr(self, 'nb_runs', 3)
        batch_size = getattr(self, 'batch_size', 32)
        
        try:
            optimized_metric_list = []
            
            for i in range(nb_runs):
                logger.info(f"Optimized run {i + 1}/{nb_runs}")
                
                metrics = await self.program.evaluate(
                    x=x_test,
                    y=y_test,
                    batch_size=batch_size,
                )
                
                optimized_metric_list.append(metrics)
                logger.info(f"Run {i + 1} metrics: {metrics}")
            
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
                title="Comparison w/o training (Math Agent with CosineSimilarity reward)",
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
        
        os.makedirs(folder, exist_ok=True)
    
    async def setup(self):
        """Complete setup process"""
        await self.setup_client()
        await self.create_agent()
        logger.info("Math agent setup completed successfully")

async def main():
    """Main execution function"""
    agent = MCPMathAgent()
    
    try:
        await agent.setup()
        
        agent.set_training_params()

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = agent.load_datasets()
        
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)
        
        # Baseline evaluation
        baseline_results = await agent.baseline_evaluation(test_data)
        
        # Training
        optimization_results = await agent.optimize_agent(train_data, val_data)
        
        # Final evaluation (after training)
        trained_results = await agent.optimized_evaluation(test_data)
        
        # Comparison plot
        agent.plot_comparison(baseline_results, trained_results)
        
        # Test query
        example_query = "What is 25 + 17 multiplied by 3?"
        result = await agent.run_query(example_query)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
