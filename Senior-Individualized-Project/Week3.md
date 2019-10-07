# SIP Journal Entry  -- Week 3

### What I worked on this week:
- I cleaned the data file in the JSON. The data is organized with six categories: Author, Description, Ingredients, Methods, (Recipe) Name, and Resource it came from. 
- I read the data from JSON to Python
```python
import pandas as pd
import json
data=pd.read_json(r"..\data\data\recipes.json",lines=True)
```
- I looked into the detailed description of Ingredients and Methods so that there's a hypothetical pattern of indicating specific recipes.
- In the end, my hypothetical categories are: pie, cake/cupcake/pancake, cookies, biscuits, rolls, scones, (possibly) brownies and tarts.


### Milestones to celebrate, or Obstacles encountered:

- After testing for the multi-layer neural network, I realized the pipeline I built is not too narrow to result in meaningful output.
So, I decided to expand the recipe particularly from cookies to entire baking in general. The topic would be Christmas recipes.
- I particulary decided to do research based on recipes from BBC website. 

### What I gained from my peer reviewer (Name):
I couldn't get a chance to meet with them since I couldn't find them in the class last Tuesday. 


### Plan for next week:
- I will be getting the probability data to see their network relationships, once again implementing the neural network and its ideal epochs to train it. 
- Possibly, I will also think about the specific data structures among graphs and how to store them -- either **breadth-first** or **depth-first** to store those outputs to visualize their relationships. 
