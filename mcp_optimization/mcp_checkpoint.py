import os
import json
import asyncio
import warnings

from synalinks.src.callbacks.program_checkpoint import ProgramCheckpoint

class MCPProgramCheckpoint(ProgramCheckpoint):
    """MCP-aware checkpoint that extends the standard ProgramCheckpoint"""
    
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
    
    def _save_program(self, epoch, batch, logs):
        """Override to use custom JSON format but preserve parent logic"""
        filepath = self._get_file_path(epoch, batch, logs)
        
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        
        current_metric = logs.get(self.monitor) if logs else None
        
        state = {
            "epoch": epoch,
            "logs": logs or {},
            "best_metric": current_metric,
            "monitor": self.monitor,
            "program_variables": self.program.get_state_tree()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _recreate_mcp_tools(self, program, mcp_client):
        """Recreate MCP tools after loading checkpoint"""
        try:
            # Get fresh tools from MCP client
            all_tools = asyncio.run(mcp_client.get_tools())
            
            # Clean tool names
            for tool in all_tools:
                tool._func.__name__ = tool._func.__name__.replace('/', '_')
            
            # Update program with fresh tools
            if hasattr(program, 'outputs'):
                program.outputs.tools = all_tools
                
        except Exception as e:
            warnings.warn(f"Failed to recreate MCP tools: {e}")
    
    def load_program_state(self, program, mcp_client=None):
        """Load checkpoint with MCP tool recreation"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {self.filepath}")
            
        with open(self.filepath, 'r') as f:
            state = json.load(f)
        
        # Restore program variables
        if "program_variables" in state:
            program.set_state_tree(state["program_variables"])
            
        # Recreate MCP tools using our built-in method
        if mcp_client:
            self._recreate_mcp_tools(program, mcp_client)
            
        return state
