# Useful debug script

uv run python debug_mixtral_mk.py

Strategy: Run each opcode stage independently — build fresh identical globals for PyVM and MK, execute only instructions up to that stage, then compare the  relevant output buffers. Stops at the first failingstage.                                                                                                   
                                                                                                                                                            
Usage:                                                                                                                                                       
# Run all stages (stops at first failure to pinpoint the  bug)                                                                                              
uv run python demos/mixtral/debug_mixtral_mkpy                                                                                                              
                                                                                                                                                            
# Test a specific stage only                                                                                                                                 
uv run python demos/mixtral/debug_mixtral_mk.py  --stageoproj                                                                                                                                               
# Run all stages without stopping (see full picture)                                                                                                         
uv run python demos/mixtral/debug_mixtral_mk.py --no-stop                                                                                                    
                                                                                                                                                            
# Just print tensor shapes/dtypes as they're passed to MK (binding mismatch check)                                                                           
uv run python demos/mixtral/debug_mixtral_mk.py --shapes-only