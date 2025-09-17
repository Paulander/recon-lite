# Testing folder 

Helper scripts to gauge performance of ReCoN implementations. 
## Features
`compare_persistent_vs_reset.md`
`performance_test.py`

Both lets you run batch runs of the networks and list specific metrics. 
As for now I have mostly used it manually to identify what to improve/states it's 
gotten stuck in (although sometimes it was better to simply look at the chessboard).

## Outlook
These could be used for an ML approach to optimize weights between nodes to improve 
the heuristics by optimizing towards the win condition. 
Due to time constraints I haven't explored this further. I want to focus on ReCoN as 
a distributed decision maker and try to make the visualizations better. 