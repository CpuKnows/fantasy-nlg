# Fantasy Sports NLG
## About
A system for hierarchical natural language generation in the domain of fantasy sports news.
Automatically generates templates from a source corpus and uses data-to-text strategies to generate summaries of a player’s performance in a game.

We gathered a parallel corpus of data and text about a particular player and game to generate these templates. We then generated text hierarchically by performing document planning, microplanning, and surface realization.

Templates improve microplanning and surface realization by maintaining syntactic and semantic structure from the source corpus. Where text aligns with data in our table we can create slots in the templates for inserting future data and generating new text. The downside of templates is that they lack variation so we break the templates from the source text into chunks which we can recombine creating a balance between these aspects.

## Dataset
We scrapped rotoworld.com for news updates about players and footballdb.com for player statistics. This allows us to construct a parallel corpus for learning data-to-text generation. 
We annotated one week of news updates as our gold corpus to compare our approaches against. We ended up with 2,118 training samples and 160 labeled test samples.

## Methods
For the document planning, microplanning, and surface realization tasks we create features following guidelines in Angeli et al. (2010) Section 3.

Several approaches to template generation were explored including: 
- MsApriori Frequent-Chunksets
- Combinatorics
- Recurrent Neural Networks
- Logistic Regression
- Naive Bayes

## Execution
### Script location: .analysis/src
- template_generation_script.py
- news_nlg_script.py

1. Execute template_generation_script.py to generate training data (templates). It
- Generates training data
- Chunks templates
- Trains the model
- Pickles model

Sample execution:

```
Processing documents to create training data
Output intermediate templates to (../data/intermediate_templates.csv)
Vectorized data shapes: (13511, 703) (13511, 21) (13511,)
Context n-gram naive bayes accuracy: 0.9818666271926578
Context tags naive bayes accuracy: 0.7793649618829103
Output templates to (../data/output_templates.csv)
Pickling template tag disambiguation model
Training chunk type selection
record (8920, 48) (8920, 1)
passing (305, 60) (305, 1)
receptions (1185, 60) (1185, 1)
rushing (518, 60) (518, 1)
game (1731, 60) (1731, 1)
Chunk type accuracy: 0.9345405724508477
Pickling template chunk type model
```

2. Execute news_nlg_script.py 
- Generates news articles.
- Default file: /data/text_output.csv'
- Sample output:

#### Artificial News Article using NLG:
Tavon Austin caught two passes for 79 yards and a touchdown in Week 2 against the Giants.

## View project presentation
https://bit.ly/2Eq9VuP

## Paper
docs/Maxwell-Singh - fantasy nlg.pdf
