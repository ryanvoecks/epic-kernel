# Project info
This repo aims to create a megakernel for mixtral which will improve performance over standard methods of running it. It uses the thunderkittens library and cuda kernels in order to achieve this. Currently it has issues and the task is to fix these issue until the megakernel can run without errors.

# Initial research

Please run `uv run python debug_mixtral_mk.py` script. This will identify the current failure point of the project which may have changed due to the work of other engineers. Then investigate the context of the repo further to understand why this issue is occuring and some likely causes.

# Targeted research

Now create debugging scripts or run commands to pinpoint the exact issue encountered and what is causing it. 

# Implement a fix

Now fix the issue based on the previous research, then test the fix using the debug script. 

# Record the findings 
If it fully passes the test explain the fix to the user. 
If it fails at the same stage iterate on different approaches, testing each time until all the initial ideas have been attempted. 
If it fails at a later stage then write to an md file a plan for future work. What was just implemented, why it improved the results and a good starting point for debugging the next issue.

# Notes
- All python scripts should be run using `uv run`
- The llama demo can be referenced as this is a working megakernel and incorporates many of the same ideas as the mixtral megakernel.